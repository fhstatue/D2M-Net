from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from model.stratified_transformer import Stratified
from model.common import MLPWithoutResidual, KPConvResBlock, AggregatorLayer, SceneContextAttentionModule

import torch_points_kernels as tp
from util.logger import get_logger
from lib.pointops2.functions import pointops
from torch_points3d.core.common_modules import FastBatchNorm1d

def compute_orthogonal_loss(F_geo, F_sem):
    """
    计算 F_geo 与 F_sem 的正交正则化损失
    输入均为已经过 L2 Normalize 的特征 (N, decoupled_dim)
    """
    cos_sim = (F_geo * F_sem).sum(dim=-1)
    ortho_loss = (cos_sim ** 2).mean()
    return ortho_loss


class D2MNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_subprototypes = args.n_subprototypes
        self.n_queries = args.n_queries
        self.n_classes = self.n_way + 1
        self.args = args
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1] + [1 for _ in range(self.n_way)]),
            ignore_index=args.ignore_label,
        )
        self.criterion_base = nn.CrossEntropyLoss(
            ignore_index=args.ignore_label
        )

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2 ** i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [
            args.patch_size * (2 ** i) for i in range(args.num_layers)
        ]
        args.quant_sizes = [
            args.quant_size * (2 ** i) for i in range(args.num_layers)
        ]

        if args.data_name == "s3dis":
            self.base_classes = 6
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    0: 1, 3: 2, 4: 3, 8: 4, 10: 5, 11: 6,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1, 2: 2, 5: 3, 6: 4, 7: 5, 9: 6,
                }
        else:  # scannet
            self.base_classes = 10
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 10: 6, 12: 7, 13: 8, 14: 9, 19: 10,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1, 4: 2, 8: 3, 9: 4, 11: 5, 15: 6, 16: 7, 17: 8, 18: 9, 20: 10,
                }

        if self.main_process():
            self.logger = get_logger(args.save_path)

        self.encoder = Stratified(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=self.base_classes + 1,  # Backbone classifier output for base classes + actual background (0)
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
            backbone=True,
            logger=get_logger(args.save_path) if self.main_process() else None,
        )

        self.feat_dim = args.channels[2]  # Main feature dimension for D2M-Net logic

        self.visualization = args.vis

        self.lin1 = nn.Sequential(
            nn.Linear(self.n_subprototypes, self.feat_dim),
            nn.ReLU(inplace=True),
        )

        self.kpconv = KPConvResBlock(
            self.feat_dim, self.feat_dim, args.grid_size * (2 ** (args.num_layers - 1 - 1)),
            # Grid size of feat_stack[-2]
            sigma=2
        )

        self.cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.feat_dim, self.n_classes),
        )
        input_dim_bk_ffn = args.channels[1] + args.channels[2]
        self.bk_ffn = nn.Sequential(
            nn.Linear(input_dim_bk_ffn, 4 * self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.feat_dim, self.feat_dim),
        )

        if self.args.data_name == "s3dis":
            agglayers = 2
        else:
            agglayers = 4

        print(f"use agglayers {agglayers}")
        self.agglayers = nn.ModuleList([
            AggregatorLayer(  # 使用已更新的AggregatorLayer
                hidden_dim=self.feat_dim, nheads=4, attention_type="linear",
            ) for _ in range(agglayers)
        ])

        if self.n_way == 1:
            self.class_reduce = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Conv1d(self.n_classes, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.class_reduce = MLPWithoutResidual(
                self.feat_dim * self.n_classes, self.feat_dim
            )

        self.bg_proto_reduce = MLPWithoutResidual(
            self.n_subprototypes * self.n_way, self.n_subprototypes
        )

        # --- 模块重构: 保留两个路径所需的所有模块 ---
        self.scene_context_encoder = SceneContextAttentionModule(
            input_dim=self.feat_dim,
            output_dim=self.feat_dim,
            num_heads=getattr(args, 'dca_scene_attn_heads', 4)
        )
        self.contextual_modulator_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim * 2, self.base_classes * self.feat_dim)
        )

        self.init_weights()

        self.register_buffer(
            "base_prototypes", torch.zeros(self.base_classes, self.feat_dim)
        )

        self.base_guidance_temp = getattr(args, 'base_guidance_temp', 1.5)
        self.modulator_gate_temp = getattr(args, 'modulator_gate_temp', 1.0)

        #########
        self.contextual_modulator_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim * 2, self.base_classes * self.feat_dim)
        )

        # ====== 任务 1 新增：几何-语义解耦分支 ======
        decoupled_dim = self.feat_dim  # 保持特征维度一致
        self.geo_branch = nn.Sequential(
            nn.Linear(self.feat_dim, decoupled_dim),
            nn.LayerNorm(decoupled_dim),  # <--- 修改点：替换为 LayerNorm
            nn.ReLU(inplace=True)
        )
        
        self.sem_branch = nn.Sequential(
            nn.Linear(self.feat_dim, decoupled_dim),
            nn.LayerNorm(decoupled_dim),  # <--- 修改点：替换为 LayerNorm
            nn.ReLU(inplace=True)
        )
        # ============================================
        
        
        # ====== 双重解耦融合门控 ======
        # 输入维度为 2（即 geo相似度 和 sem相似度），输出融合权重 alpha
        self.dual_fusion_gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # ============================================

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_parameters():
            # 确保 "confidence_scorer" 从特殊初始化列表中移除
            if any(n_part in name for n_part in [
                "class_attention.gate_mlp",  # gate_mlp_context, gate_mlp_pointwise
                "class_attention.bg_guidance_project",
                "class_attention.strategy_mixing_gate_mlp",  # <--- 新的门控MLP
                "scene_context_encoder", "contextual_modulator_mlp"
            ]):
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)  # Init bias to zero for these specific layers
                if m.dim() > 1 and not isinstance(m, nn.LayerNorm) and not isinstance(m, FastBatchNorm1d):
                    nn.init.xavier_uniform_(m)
                continue  # Skip generic init for these, allow custom or default for submodules.

            if m.dim() > 1:  # For other layers
                nn.init.xavier_uniform_(m)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, FastBatchNorm1d):  # Handle norms if needed
                if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
                if hasattr(m, 'weight') and m.weight is not None: nn.init.ones_(m.weight)

    def main_process(self):
        return not self.args.multiprocessing_distributed or (
                self.args.multiprocessing_distributed
                and self.args.rank % self.args.ngpus_per_node == 0
        )

    def forward(
            self,
            support_offset: torch.Tensor,
            support_x: torch.Tensor,
            support_y: torch.Tensor,
            query_offset: torch.Tensor,
            query_x: torch.Tensor,
            query_y: torch.Tensor,
            epoch: int,
            support_base_y: Optional[torch.Tensor] = None,
            query_base_y: Optional[torch.Tensor] = None,
            sampled_classes: Optional[np.array] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # get downsampled support features
        (
            support_feat_full,
            support_x_low,
            support_offset_low,
            support_y_low,
            _,
            support_base_y_low,
        ) = self.getFeatures(
            support_x, support_offset, support_y, support_base_y
        )
        assert support_y_low.shape[0] == support_x_low.shape[0]

        support_offset_low_cpu = support_offset_low[:-1].long().cpu()
        support_feat_list = torch.tensor_split(support_feat_full, support_offset_low_cpu)
        support_x_low_list = torch.tensor_split(support_x_low, support_offset_low_cpu)

        support_base_y_list = None
        if support_base_y_low is not None:
            support_base_y_list = torch.tensor_split(
                support_base_y_low, support_offset_low_cpu
            )

        # get prototypes
        fg_mask = support_y_low
        bg_mask = torch.logical_not(support_y_low)
        fg_mask_list = torch.tensor_split(fg_mask, support_offset_low_cpu)
        bg_mask_list = torch.tensor_split(bg_mask, support_offset_low_cpu)

        fg_prototypes = self.getPrototypes(
            support_x_low_list,
            support_feat_list,
            fg_mask_list,
            k=self.n_subprototypes // self.k_shot,
        )
        bg_prototype = self.getPrototypes(
            support_x_low_list,
            support_feat_list,
            bg_mask_list,
            k=self.n_subprototypes // self.k_shot,
        )

        if bg_prototype.shape[0] > self.n_subprototypes:
            bg_prototype = self.bg_proto_reduce(
                bg_prototype.permute(1, 0)
            ).permute(1, 0)

        sparse_embeddings = torch.cat(
            [bg_prototype, fg_prototypes]
        )

         # ====== 方案三新增：对 Support 原型（Prototypes）也进行解耦 ======
        Proto_geo = F.normalize(self.geo_branch(sparse_embeddings), p=2, dim=-1)
        Proto_sem = F.normalize(self.sem_branch(sparse_embeddings), p=2, dim=-1)
        # =================================================================

        # get downsampled query features
        (
            query_feat_full,
            query_x_low,
            query_offset_low,
            query_y_low,
            q_base_pred,
            query_base_y_low,
        ) = self.getFeatures(query_x, query_offset, query_y, query_base_y)

        query_offset_low_cpu = query_offset_low[:-1].long().cpu()
        query_feat_list = torch.tensor_split(query_feat_full, query_offset_low_cpu)
        query_x_low_list_for_kpconv = torch.tensor_split(
            query_x_low, query_offset_low_cpu
        )

        query_base_y_low_list = None
        if query_base_y_low is not None:
            query_base_y_low_list = torch.tensor_split(
                query_base_y_low, query_offset_low_cpu
            )

        if self.training:
            combined_feats_for_base_update = list(query_feat_list) + list(support_feat_list)
            combined_base_labels_for_update = []
            if query_base_y_low_list is not None: combined_base_labels_for_update.extend(list(query_base_y_low_list))
            if support_base_y_list is not None: combined_base_labels_for_update.extend(list(support_base_y_list))

            if combined_base_labels_for_update:
                for scene_idx_update in range(len(combined_feats_for_base_update)):
                    base_feat_update = combined_feats_for_base_update[scene_idx_update]
                    base_y_update = combined_base_labels_for_update[scene_idx_update]
                    cur_baseclsses = base_y_update.unique()
                    cur_baseclsses = cur_baseclsses[cur_baseclsses != self.args.ignore_label]
                    cur_baseclsses = cur_baseclsses[cur_baseclsses != 0]
                    for class_label_val in cur_baseclsses:
                        if class_label_val.item() > 0 and class_label_val.item() <= self.base_classes:
                            proto_idx_for_buffer = class_label_val.item() - 1
                            class_mask_update = (base_y_update == class_label_val)
                            if class_mask_update.sum() > 0:
                                class_features_update = (base_feat_update[class_mask_update].sum(
                                    dim=0) / class_mask_update.sum()).detach()
                                if torch.all(self.base_prototypes[proto_idx_for_buffer] == 0):
                                    self.base_prototypes[proto_idx_for_buffer] = class_features_update
                                else:
                                    self.base_prototypes[proto_idx_for_buffer] = (self.base_prototypes[
                                                                                      proto_idx_for_buffer] * 0.995 + class_features_update * 0.005)

        # --- Part 2: Prepare global available base prototypes (BPC Part 2: Filtering) ---
        # 这个块是必须的，因为它定义了 'global_base_avail_pts'

        global_base_protos_for_ep = self.base_prototypes.clone()
        base_filter_mask = torch.ones(self.base_prototypes.shape[0], dtype=torch.bool,
                                      device=self.base_prototypes.device)  # 默认为全可用

        if self.training:
            indices_to_mask_in_buffer = []
            if sampled_classes is not None:
                for novel_cls_original_label in sampled_classes:
                    if novel_cls_original_label in self.base_class_to_pred_label:
                        internal_base_id_1_indexed = self.base_class_to_pred_label[novel_cls_original_label]
                        buffer_idx_to_mask = internal_base_id_1_indexed - 1
                        indices_to_mask_in_buffer.append(buffer_idx_to_mask)

            if indices_to_mask_in_buffer:
                # 只在有需要mask的索引时才更新mask
                base_filter_mask[torch.tensor(indices_to_mask_in_buffer, device=base_filter_mask.device).long()] = False

        # 无论是训练还是评估，都使用mask来获取可用的原型
        # 在评估时，因为 indices_to_mask_in_buffer 为空, base_filter_mask 会全为True
        global_base_avail_pts = global_base_protos_for_ep[base_filter_mask]

        # --- End of Part 2 ---

        # ========================================================
        Proto_geo = F.normalize(self.geo_branch(sparse_embeddings), p=2, dim=-1)
        Proto_sem = F.normalize(self.sem_branch(sparse_embeddings), p=2, dim=-1)
        
        query_pred_final_list = []
        total_ortho_loss = 0.0  
        # =========================================================================

        for scene_idx, q_feat_scene in enumerate(query_feat_list):

            # ====== 1. 对 Query 特征进行解耦 (我帮你清理了重复代码，统一叫 Q_geo 和 Q_sem) ======
            Q_geo = F.normalize(self.geo_branch(q_feat_scene), p=2, dim=-1)
            Q_sem = F.normalize(self.sem_branch(q_feat_scene), p=2, dim=-1)
            
            # 累加正交正则化损失
            total_ortho_loss += compute_orthogonal_loss(Q_geo, Q_sem)
            # =========================================================================

            # --- 全新的、完全自适应的BPC逻辑 (保持不变) ---
            scene_context_feat_enhanced = None
            refined_base_guidance_scene = None

            if epoch >= 1:  # Warmup
                # 1. 计算路径A的引导信息 (全局上下文)
                scene_context_feat_enhanced = self.scene_context_encoder(q_feat_scene.unsqueeze(0)).squeeze(0)

                # 2. 计算路径B的引导信息 (逐点引导)
                if global_base_avail_pts.shape[0] > 0:
                    # 使用 scene_context_feat_enhanced (或者简单的torch.mean) 来驱动调制器
                    all_base_modulation_params = self.contextual_modulator_mlp(scene_context_feat_enhanced)
                    all_base_modulation_params_reshaped = all_base_modulation_params.view(self.base_classes,
                                                                                          self.feat_dim)

                    selected_modulation_params = all_base_modulation_params_reshaped[base_filter_mask]

                    dynamic_context_aware_protos = global_base_avail_pts
                    if selected_modulation_params.shape[0] == global_base_avail_pts.shape[0]:
                        processed_gates = torch.sigmoid(selected_modulation_params / self.modulator_gate_temp)
                        dynamic_context_aware_protos = global_base_avail_pts * processed_gates

                    base_similarity_raw = F.cosine_similarity(
                        q_feat_scene.unsqueeze(1), dynamic_context_aware_protos.unsqueeze(0), dim=2)
                    attention_weights = F.softmax(base_similarity_raw / self.base_guidance_temp, dim=1)
                    refined_base_guidance_scene = (attention_weights * base_similarity_raw).sum(dim=1, keepdim=True)

            # ====== 深度解耦与双重自适应匹配 (替换了旧代码的 CMC) ======
            # 1. 分别在几何与语义子空间计算 Query 与 Support原型的相似度
            Corr_geo = F.cosine_similarity(Q_geo.unsqueeze(1), Proto_geo.unsqueeze(0), dim=2) # [N_points, N_protos]
            Corr_sem = F.cosine_similarity(Q_sem.unsqueeze(1), Proto_sem.unsqueeze(0), dim=2) # [N_points, N_protos]

            # 2. 拼接双相似度，经过 dual_fusion_gate 预测自适应融合权重 alpha
            corr_cat = torch.stack([Corr_geo, Corr_sem], dim=-1) # [N_points, N_protos, 2]
            alpha = self.dual_fusion_gate(corr_cat).squeeze(-1)  # [N_points, N_protos]

            # 3. 计算最终的自适应加权相关性得分
            correlations_scene = alpha * Corr_geo + (1 - alpha) * Corr_sem

            # 4. 保持原来的维度变换，准备送给后续聚合网络
            correlations_scene = self.lin1(
                correlations_scene.view(correlations_scene.shape[0], self.n_way + 1, -1)).permute(0, 2, 1)
            correlations_scene_for_agg = correlations_scene.permute(1, 2, 0).unsqueeze(0)
            # ======================================================================

            # 3. HCA调用，传递所有的信息供内部混合
            temp_correlations_agg = correlations_scene_for_agg
            for layer in self.agglayers:
                temp_correlations_agg = layer(
                    temp_correlations_agg,
                    scene_context_feat_enhanced,  # 路径A信息
                    refined_base_guidance_scene,  # 路径B信息
                )

            # --- Part 4: Decoding and Prediction (Unchanged) ---
            correlations_after_agg = temp_correlations_agg.squeeze(0).permute(2, 1, 0).contiguous()

            # Reduce class dimension
            if self.n_way == 1:
                processed_correlations = self.class_reduce(correlations_after_agg).squeeze(1)
            else:
                processed_correlations = self.class_reduce(
                    correlations_after_agg.reshape(correlations_after_agg.shape[0], -1)
                )

            # KPConv layer
            coord_scene = query_x_low_list_for_kpconv[scene_idx]
            batch_scene = torch.zeros(processed_correlations.shape[0], dtype=torch.int64, device=coord_scene.device)
            radius_kp = 2.5 * self.args.grid_size * (2 ** (self.args.num_layers - 2)) * 2.0
            neighbors_scene = \
                tp.ball_query(radius_kp, self.args.max_num_neighbors, coord_scene, coord_scene, mode="partial_dense",
                              batch_x=batch_scene, batch_y=batch_scene)[0]

            final_scene_feat = self.kpconv(processed_correlations, coord_scene, batch_scene, neighbors_scene.clone())

            # Classification layer
            out_scene = self.cls(final_scene_feat)
            query_pred_final_list.append(out_scene)

        query_pred_batch = torch.cat(query_pred_final_list, dim=0)

        assert not torch.any(torch.isnan(query_pred_batch)), "NaN found in query_pred_batch"

         # --- 任务 2 结算: 将 Ortho Loss 以 lambda=0.1 加入最终损失 ---
        loss = self.criterion(query_pred_batch, query_y_low)
        avg_ortho_loss = total_ortho_loss / len(query_feat_list) if len(query_feat_list) > 0 else 0.0
        # loss += 0.1 * avg_ortho_loss
        
        # 建议修改为：
        lambda_ortho = getattr(self.args, 'lambda_ortho', 0.1)
        loss += lambda_ortho * avg_ortho_loss

        if query_base_y_low is not None and q_base_pred is not None:
            loss += self.criterion_base(q_base_pred, query_base_y_low.to(q_base_pred.device))

        final_pred_interpolated = pointops.interpolation(
            query_x_low,
            query_x[:, :3].contiguous().cuda(),
            query_pred_batch.contiguous(),
            query_offset_low.cuda(),
            query_offset.cuda(),
        ).transpose(0, 1).unsqueeze(0)

        if self.visualization and self.main_process():
            self.vis(
                query_offset, query_x, query_y,
                support_offset, support_x, support_y,
                final_pred_interpolated,
            )

        return final_pred_interpolated, loss

    def getFeatures(self, ptclouds, offset, gt, query_base_y=None):
        coord, feat_color = (
            ptclouds[:, :3].contiguous(),
            ptclouds[:, 3:6].contiguous(),
        )

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        radius_stem = 2.5 * self.args.grid_size * 1.0
        batch = batch.to(coord.device)
        neighbor_idx_stem = tp.ball_query(
            radius_stem,
            self.args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[0]

        coord = coord.cuda(non_blocking=True)
        feat_color = feat_color.cuda(non_blocking=True)
        offset_gpu = offset.cuda(non_blocking=True)
        gt_gpu = gt.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx_stem = neighbor_idx_stem.cuda(non_blocking=True)

        input_feat_to_encoder = feat_color
        if self.args.concat_xyz:
            input_feat_to_encoder = torch.cat([feat_color, coord], 1)

        query_base_y_gpu = None
        if query_base_y is not None:
            query_base_y_gpu = query_base_y.cuda(non_blocking=True)

        (
            feat_from_encoder,
            coord_from_encoder,
            offset_from_encoder,
            gt_downsampled,
            base_pred_from_encoder,
            query_base_y_downsampled
        ) = self.encoder(
            input_feat_to_encoder, coord, offset_gpu, batch, neighbor_idx_stem, gt_gpu, query_base_y_gpu
        )
        final_feat = self.bk_ffn(feat_from_encoder)
        return final_feat, coord_from_encoder, offset_from_encoder, gt_downsampled, base_pred_from_encoder, query_base_y_downsampled

    def getPrototypes(self, coords_list, feats_list, masks_list, k=100):
        prototypes_all_shots = []
        num_support_scenes = len(coords_list)

        for i in range(num_support_scenes):
            coord_scene = coords_list[i][:, :3]
            feat_scene = feats_list[i]
            mask_scene = masks_list[i].bool()

            if mask_scene.sum() == 0:
                zero_protos = feat_scene.new_zeros(k, self.feat_dim)
                prototypes_all_shots.append(zero_protos)
                if self.main_process() and hasattr(self, 'logger'):
                    self.logger.warning(f"Shot {i}: No points for prototype generation. Appending zero prototypes.")
                continue
            coord_masked = coord_scene[mask_scene]
            feat_masked = feat_scene[mask_scene]
            if coord_masked.shape[0] == 0:
                zero_protos = feat_scene.new_zeros(k, self.feat_dim)
                prototypes_all_shots.append(zero_protos)
                if self.main_process() and hasattr(self, 'logger'):
                    self.logger.warning(f"Shot {i}: coord_masked empty. Appending zero protos.")
                continue
            protos_one_shot = self.getMutiplePrototypes(coord_masked, feat_masked, k)
            prototypes_all_shots.append(protos_one_shot)

        if not prototypes_all_shots:
            return torch.empty(0, self.feat_dim, device=self.base_prototypes.device)
        return torch.cat(prototypes_all_shots, dim=0)

    def getMutiplePrototypes(self, coord, feat, num_prototypes_to_extract):
        num_available_points = feat.shape[0]
        if num_available_points == 0:
            return feat.new_zeros(num_prototypes_to_extract, self.feat_dim)

        if num_available_points <= num_prototypes_to_extract:
            padding_needed = num_prototypes_to_extract - num_available_points
            padded_feats = feat
            if padding_needed > 0:
                zero_padding = feat.new_zeros(padding_needed, self.feat_dim)
                padded_feats = torch.cat([feat, zero_padding], dim=0)
            return padded_feats

        if coord.shape[0] < 1:
            return feat.new_zeros(num_prototypes_to_extract, self.feat_dim)

        fps_indices = pointops.furthestsampling(
            coord.contiguous(),
            torch.cuda.IntTensor([coord.shape[0]]),
            torch.cuda.IntTensor([num_prototypes_to_extract]),
        ).long()
        actual_num_fps_seeds = fps_indices.shape[0]

        if actual_num_fps_seeds == 0:
            return feat.new_zeros(num_prototypes_to_extract, self.feat_dim)

        farthest_seeds_features = feat[fps_indices]
        distances_to_seeds = torch.cdist(feat, farthest_seeds_features)
        assignments_to_seeds = torch.argmin(distances_to_seeds, dim=1)
        prototypes_final = feat.new_zeros(actual_num_fps_seeds, self.feat_dim)

        for i in range(actual_num_fps_seeds):
            points_in_cluster_indices = (assignments_to_seeds == i).nonzero(as_tuple=True)[0]
            if len(points_in_cluster_indices) == 0:
                prototypes_final[i] = farthest_seeds_features[i]
                if self.main_process() and hasattr(self, 'logger'):
                    self.logger.info(f"Empty cluster for seed {i}. Using FPS seed.")
            else:
                prototypes_final[i] = feat[points_in_cluster_indices].mean(dim=0)

        if actual_num_fps_seeds < num_prototypes_to_extract:
            padding_needed = num_prototypes_to_extract - actual_num_fps_seeds
            zero_padding = feat.new_zeros(padding_needed, self.feat_dim)
            prototypes_final = torch.cat([prototypes_final, zero_padding], dim=0)
        return prototypes_final

    def vis(
            self,
            query_offset, query_x, query_y,
            support_offset, support_x, support_y,
            final_pred,
    ):
        query_offset_cpu = query_offset.long().cpu()
        query_x_cpu = query_x.cpu()
        query_y_cpu = query_y.cpu()
        final_pred_cpu = final_pred.squeeze(0).max(0)[1].cpu()
        support_offset_cpu = support_offset.long().cpu()
        support_x_cpu = support_x.cpu()
        support_y_cpu = support_y.cpu()

        sp_nps_vis, sp_fgs_vis = [], []
        start_idx_sp = 0
        for i in range(support_offset_cpu.shape[0] - 1):
            end_idx_sp = support_offset_cpu[i + 1]
            support_x_scene = support_x_cpu[start_idx_sp:end_idx_sp]
            support_y_scene = support_y_cpu[start_idx_sp:end_idx_sp]
            start_idx_sp = end_idx_sp
            sp_np_scene = support_x_scene.numpy().copy()
            sp_np_scene[:, 3:6] = sp_np_scene[:, 3:6] * 255.0
            sp_nps_vis.append(wandb.Object3D(sp_np_scene))
            sp_fg_scene_data = np.concatenate((sp_np_scene[:, :3], support_y_scene.unsqueeze(-1).numpy()), axis=-1)
            sp_fgs_vis.append(wandb.Object3D(sp_fg_scene_data))

        qu_s_vis, qu_gts_vis, qu_pds_vis = [], []
        start_idx_qu = 0
        for i in range(query_offset_cpu.shape[0] - 1):
            end_idx_qu = query_offset_cpu[i + 1]
            query_x_scene = query_x_cpu[start_idx_qu:end_idx_qu]
            query_y_scene_orig = query_y_cpu[start_idx_qu:end_idx_qu]
            pred_scene = final_pred_cpu[start_idx_qu:end_idx_qu]
            start_idx_qu = end_idx_qu
            qu_scene_np = query_x_scene.numpy().copy()
            qu_scene_np[:, 3:6] = qu_scene_np[:, 3:6] * 255.0
            qu_s_vis.append(wandb.Object3D(qu_scene_np))
            query_y_scene_vis = torch.where(query_y_scene_orig == self.args.ignore_label,
                                            torch.tensor(0, dtype=query_y_scene_orig.dtype,
                                                         device=query_y_scene_orig.device), query_y_scene_orig)
            qu_gt_scene_data = np.concatenate((qu_scene_np[:, :3], query_y_scene_vis.unsqueeze(-1).numpy()), axis=-1)
            qu_gts_vis.append(wandb.Object3D(qu_gt_scene_data))
            qu_pd_scene_data = np.concatenate((qu_scene_np[:, :3], pred_scene.unsqueeze(-1).numpy()), axis=-1)
            qu_pds_vis.append(wandb.Object3D(qu_pd_scene_data))

        wandb.log({
            "Support_Points": sp_nps_vis, "Support_Masks": sp_fgs_vis,
            "Query_Points": qu_s_vis, "Query_Preds": qu_pds_vis, "Query_GTs": qu_gts_vis,
        })
