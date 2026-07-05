import torch
import torch.nn as nn

from einops import rearrange

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv.kernels import KPConvLayer


class SceneContextAttentionModule(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout,
                                               batch_first=True)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_scene_features):
        attended_features, _ = self.attention(x_scene_features, x_scene_features, x_scene_features)
        x_scene_features = self.norm(x_scene_features + attended_features)
        pooled_context = torch.mean(x_scene_features, dim=1)
        scene_context_vector = self.fc_out(self.act(pooled_context))
        return scene_context_vector


class KPConvResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            prev_grid_size,
            sigma=1.0,
            negative_slope=0.2,
            bn_momentum=0.02,
    ):
        super().__init__()
        d_2 = out_channels // 4
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.unary_1 = torch.nn.Sequential(
            nn.Linear(in_channels, d_2, bias=False),
            FastBatchNorm1d(d_2, momentum=bn_momentum),
            activation,
        )
        self.unary_2 = torch.nn.Sequential(
            nn.Linear(d_2, out_channels, bias=False),
            FastBatchNorm1d(out_channels, momentum=bn_momentum),
            activation,
        )
        self.kpconv = KPConvLayer(
            d_2, d_2, point_influence=prev_grid_size * sigma, add_one=False
        )
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = activation

        if in_channels != out_channels:
            self.shortcut_op = torch.nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                FastBatchNorm1d(out_channels, momentum=bn_momentum),
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]
        shortcut = feats.clone()
        feats = self.unary_1(feats)
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.unary_2(feats)
        shortcut = self.shortcut_op(shortcut)
        feats = feats + shortcut
        return feats


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]),
                float("-inf"),
            )

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = (
                torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        )

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(
            self, hidden_dim, guidance_dim, nheads=8, attention_type="linear"
    ):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == "linear":
            self.attention = LinearAttention()
        elif attention_type == "full":
            self.attention = FullAttention()
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Arguments:
            x: B, L, C
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.nheads)
        k = rearrange(k, "B S (H D) -> B S H D", H=self.nheads)
        v = rearrange(v, "B S (H D) -> B S H D", H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, "B L H D -> B L (H D)")
        return out


# --- 主要修改 ClassTransformerLayer，引入逐点策略混合门控 ---
class ClassTransformerLayer(nn.Module):
    def __init__(
            self,
            hidden_dim=64,
            nheads=8,
            attention_type="linear",
    ) -> None:
        super().__init__()
        # 类别相关性之间的注意力保持不变
        self.attention = AttentionLayer(
            hidden_dim=hidden_dim, guidance_dim=0, nheads=nheads, attention_type=attention_type
        )
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # --- 双路径校准机制 ---
        # 路径 A: 基于全局上下文的门控
        self.gate_mlp_context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        # 路径 B: 基于逐点引导的门控
        self.bg_point_guidance_project = nn.Linear(1, hidden_dim)
        self.gate_mlp_pointwise = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )

        # --- 全新模块: 逐点策略混合门控 (Point-wise Strategy-Mixing Gate) ---
        # F_geo 和 F_sem 拼接后维度为 hidden_dim * 2
        self.strategy_mixing_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        

    # --- ClassTransformerLayer 全新的 forward 签名 ---
    def forward(self, x, scene_context_feat, point_base_guidance, F_geo=None, F_sem=None):
        """
        参数:
            x (Tensor): 相关性特征 -> (B, C, T, N)
            scene_context_feat (Tensor | None): 增强的场景上下文特征 -> (C)
            point_base_guidance (Tensor | None): 逐点的基类引导信号 -> (N, 1)
        """
        B, C_dim, T_classes, N_points = x.size()
        x_pool = rearrange(x, "B C T N -> (B N) T C")

        idx_background = 0
        original_bg_corr = x_pool[:, idx_background].clone()

        # 1. 计算路径A的校准结果 (Context-driven)
        calibrated_bg_context = original_bg_corr  # 默认不校准
        if scene_context_feat is not None:
            expanded_context = scene_context_feat.expand_as(original_bg_corr)
            gate_input_context = torch.cat([original_bg_corr, expanded_context], dim=1)
            gate_context = self.gate_mlp_context(gate_input_context)
            calibrated_bg_context = original_bg_corr * (1 - gate_context)

        # 2. 计算路径B的校准结果 (Point-wise driven)
        calibrated_bg_pointwise = original_bg_corr  # 默认不校准
        projected_point_guidance = None
        if point_base_guidance is not None:
            projected_point_guidance = self.bg_point_guidance_project(point_base_guidance)
            gate_input_pointwise = torch.cat([original_bg_corr, projected_point_guidance], dim=1)
            gate_pointwise = self.gate_mlp_pointwise(gate_input_pointwise)
            calibrated_bg_pointwise = original_bg_corr * (1 - gate_pointwise)

        # 3. 计算逐点混合权重 lambda_pointwise
        # 这个权重决定了每个点在多大程度上相信“全局上下文路径”
        lambda_pointwise = torch.full_like(original_bg_corr[:, :1], 0.5)  # 默认中立权重
        if projected_point_guidance is not None:
            gate_input_for_mixing = torch.cat([original_bg_corr.detach(), projected_point_guidance], dim=1)
            lambda_pointwise = self.strategy_mixing_gate_mlp(gate_input_for_mixing)

        # 4. 使用逐点的 lambda_pointwise 融合两条路径的校准结果
        # lambda_pointwise 接近1 -> 更相信全局上下文校准
        # lambda_pointwise 接近0 -> 更相信逐点引导校准
        final_calibrated_bg_corr = lambda_pointwise * calibrated_bg_context + \
                                   (1 - lambda_pointwise) * calibrated_bg_pointwise

        # 将最终校准结果应用回特征池
        x_pool_calibrated = x_pool.clone()
        x_pool_calibrated[:, idx_background] = final_calibrated_bg_corr

        # 后续的注意力、FFN和残差连接
        x_pool_after_attention = self.attention(self.norm1(x_pool_calibrated))
        x_pool = x_pool_calibrated + x_pool_after_attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool))
        x_pool = rearrange(x_pool, "(B N) T C -> B C T N", B=B, N=N_points)
        x = x + x_pool
        return x


class SpatialTransformerLayer(nn.Module):
    def __init__(
            self,
            hidden_dim=64,
            guidance_dim=64,
            nheads=8,
            attention_type="linear",
    ) -> None:
        super().__init__()
        self.attention = AttentionLayer(
            hidden_dim,
            guidance_dim,
            nheads=nheads,
            attention_type=attention_type,
        )
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Arguments:
            x: B, C, T, N
        """
        B, _, T, N = x.size()

        x_pool = rearrange(x, "B C T N -> (B T) N C")

        x_pool = x_pool + self.attention(self.norm1(x_pool))  # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool))  # MLP

        x_pool = rearrange(x_pool, "(B T) N C -> B C T N", T=T)

        x = x + x_pool  # Residual
        return x


# --- 修改 AggregatorLayer ---
class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim=64, nheads=4, attention_type="linear") -> None:
        super().__init__()
        self.spatial_attention = SpatialTransformerLayer(hidden_dim, 0, nheads, attention_type)
        self.class_attention = ClassTransformerLayer(hidden_dim, nheads, attention_type)

    def forward(self, x, scene_context_feat, point_base_guidance):
        x = self.spatial_attention(x)
        x = self.class_attention(x, scene_context_feat, point_base_guidance)
        return x


class MLPWithoutResidual(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4 * hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
