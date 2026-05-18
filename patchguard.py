import re
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from spectral_filter import MultiLayerSpectralFilter

def my_forward_wrapper(attn_obj):
    def my_forward(x, attn_mask=None):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x

    return my_forward

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module for combining features from different ViT layers.
    Includes a 'gated_attention' mode specifically designed for robust anomaly detection.
    """
    def __init__(self, in_dims, out_dim=None, fusion_type='concat_proj'):
        """
        Args:
            in_dims: list of input dimensions from each layer (e.g., [384, 384, 768])
                     Order should be [shallow_1, shallow_2, ..., deep_final]
            out_dim: output dimension (required for 'concat_proj' and 'attention')
            fusion_type: 'concat' | 'concat_proj' | 'scalar_attention' | 'gated_attention'
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.in_dims = in_dims
        self.num_scales = len(in_dims)
        
        # Determine output dimension
        self.out_dim = out_dim if out_dim else in_dims[-1]  # Default to last layer's dim

        if fusion_type == 'concat':
            self.out_dim = sum(in_dims)
            
        elif fusion_type == 'concat_proj':
            self.proj = nn.Linear(sum(in_dims), self.out_dim)
            self.norm = nn.LayerNorm(self.out_dim)
            # Optional: Add dropout for regularization
            self.dropout = nn.Dropout(0.1)
            
        elif fusion_type == 'scalar_attention':
            # Project all inputs to the same dimension first
            self.scale_projs = nn.ModuleList([
                nn.Linear(d, self.out_dim) for d in in_dims
            ])
            # Learnable scalar weights for each scale (initialized to 0 -> softmax makes them equal)
            self.scale_weights = nn.Parameter(torch.zeros(self.num_scales))
            self.norm = nn.LayerNorm(self.out_dim)

        elif fusion_type == 'gated_attention':
            """
            Advanced mode: Uses the Deepest Layer (presumed most robust) to 
            generate spatial attention maps for Shallow Layers.
            """
            # Project all inputs to output dimension
            self.scale_projs = nn.ModuleList([
                nn.Linear(d, self.out_dim) for d in in_dims
            ])
            
            # Spatial Gate: Compresses concatenated features to generate per-token weights
            # Input: concatenated features -> Output: num_scales weights per token
            self.gate_net = nn.Sequential(
                nn.Linear(self.out_dim * self.num_scales, self.out_dim // 2),
                nn.GELU(),
                nn.Linear(self.out_dim // 2, self.num_scales)
            )
            self.norm = nn.LayerNorm(self.out_dim)

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
            
        # Initialize weights (Good practice for ViT components)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features_list, return_weights=False):
        """
        Args:
            features_list: list of tensors, each [B, N, D_i]
                           Assumption: N (num_patches) is consistent across layers.
            return_weights: if True and fusion_type is 'gated_attention',
                            also return the spatial_weights [B, N, num_scales]
        """
        # Checks
        assert len(features_list) == self.num_scales, f"Expected {self.num_scales} inputs, got {len(features_list)}"

        if self.fusion_type == 'concat':
            return (torch.cat(features_list, dim=-1), None) if return_weights else torch.cat(features_list, dim=-1)
        
        elif self.fusion_type == 'concat_proj':
            concatenated = torch.cat(features_list, dim=-1)
            projected = self.proj(concatenated)
            result = self.norm(self.dropout(projected))
            return (result, None) if return_weights else result
        
        elif self.fusion_type == 'scalar_attention':
            # Softmax over the scale dimension
            weights = F.softmax(self.scale_weights, dim=0) # [num_scales]
            
            # Project and Weighted Sum
            fused = 0
            for i, (proj, feat) in enumerate(zip(self.scale_projs, features_list)):
                fused += weights[i] * proj(feat)
                
            result = self.norm(fused)
            return (result, weights.unsqueeze(0).unsqueeze(0).expand(features_list[0].shape[0], features_list[0].shape[1], -1)) if return_weights else result

        elif self.fusion_type == 'gated_attention':
            # 1. Project all features to the same dimension [B, N, out_dim]
            projected_feats = [proj(f) for proj, f in zip(self.scale_projs, features_list)]
            
            # 2. Stack them to calculate gating weights [B, N, num_scales * out_dim]
            concat_feats = torch.cat(projected_feats, dim=-1)
            
            # 3. Calculate Spatial Weights: [B, N, num_scales]
            # This allows the model to say: "For Patch 5, trust Layer 10; for Patch 100, trust Layer 3"
            raw_scores = self.gate_net(concat_feats)
            spatial_weights = F.softmax(raw_scores, dim=-1) 
            
            # 4. Weighted Sum
            fused = torch.zeros_like(projected_feats[0])
            for i in range(self.num_scales):
                # weights[:, :, i:i+1] -> [B, N, 1] for broadcasting
                w = spatial_weights[:, :, i:i+1]
                fused += w * projected_feats[i]
                
            result = self.norm(fused)
            return (result, spatial_weights) if return_weights else result

class FeatureExtractor(nn.Module):
    def __init__(self, hf_path, feature_layer_indices, reg_layer_indices, image_size, device, 
                 fusion_type='concat', fusion_dim=None,
                 spectral_filter_layers=None, spectral_gate_reduction=4,
                 spectral_reg_type='l1', spectral_reg_weight=0.01,
                 spectral_strength=0.1,
                 use_multiscale_fusion=True, use_spectral_filter=True):
        super(FeatureExtractor, self).__init__()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mu = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)
        self.norm = lambda x: (x - self.mu) / self.std

        self.use_multiscale_fusion = use_multiscale_fusion
        self.use_spectral_filter = use_spectral_filter

        self.feature_layer_indices = feature_layer_indices
        self.reg_layer_indices = reg_layer_indices

        self.pretrained_model = timm.create_model(hf_path, pretrained=False, num_classes=0, img_size=image_size).to(device)

        self.base_embed_dim = self.pretrained_model.embed_dim
        self.patch_size = self.pretrained_model.patch_embed.patch_size[0]
        self.num_patches = (image_size // self.patch_size) ** 2

        # 多层融合模块 (Multi-scale Fusion) — 即插即用
        if self.use_multiscale_fusion:
            in_dims = [self.base_embed_dim] * len(feature_layer_indices)
            self.fusion = MultiScaleFeatureFusion(in_dims, fusion_dim, fusion_type).to(device)
            self.embed_dim = self.fusion.out_dim
        else:
            self.fusion = None
            self.embed_dim = self.base_embed_dim  # 退化为单层特征维度

        pattern = r'reg(\d+)'
        match = re.search(pattern, hf_path)
        self.start_index = int(match.group(1)) + 1 if match else 1

        indices = set(feature_layer_indices + reg_layer_indices)
        for i in indices:
            self.pretrained_model.blocks[i-1].attn.forward = my_forward_wrapper(self.pretrained_model.blocks[i-1].attn)
        
        # 频域过滤模块 (Spectral Filtering) — 即插即用
        self.spectral_filter_layers = spectral_filter_layers or []
        self.spectral_filter = None
        if self.use_spectral_filter and self.spectral_filter_layers:
            self.spectral_filter = MultiLayerSpectralFilter(
                layer_indices=self.spectral_filter_layers,
                embed_dim=self.base_embed_dim,
                num_patches=self.num_patches,
                gate_reduction=spectral_gate_reduction,
                regularization=spectral_reg_type,
                reg_weight=spectral_reg_weight,
                spectral_strength=spectral_strength,
                device=device
            )
            
    def forward(self, x, use_reg=True, return_spectral_info=False, return_fusion_weights=False, bypass_spectral=False):
        x = self.norm(x)
        x = self.pretrained_model.patch_embed(x)
        x = self.pretrained_model._pos_embed(x)
        x = self.pretrained_model.patch_drop(x)
        x = self.pretrained_model.norm_pre(x)

        out = []
        attention_weights = []
        spectral_info = {}  # {layer_idx: {'before': tensor, 'after': tensor}}

        current_batch_spectral_loss = 0.0
        # iterating through the layers up to last layer to extract from
        for idx, layer in enumerate(self.pretrained_model.blocks, start=1):
            x = layer(x)

            # ❶ 先提取特征（从干净的 x）—— 保证 fusion 收到未被频域过滤扭曲的特征
            if idx in self.feature_layer_indices:
                features_layer = self.pretrained_model.norm(x[:, self.start_index:, :])
                
                if self.use_multiscale_fusion:
                    # 多层融合模式：收集所有层的特征
                    out.append(features_layer)
                else:
                    # 单层模式：只保留最后一层（最深层）的特征
                    out = [features_layer]

            # ❷ 再频域过滤（修改 x）—— 过滤效果传播到所有后续层，惠及深层特征
            # bypass_spectral=True 时跳过（训练 PGD 生成阶段使用，防止攻击者适应过滤器）
            if not bypass_spectral and self.use_spectral_filter and self.spectral_filter is not None and idx in self.spectral_filter:
                x_patches = x[:, self.start_index:, :]
                
                if return_spectral_info:
                    spectral_info[idx] = {'before': x_patches.detach().clone()}
                
                x_filtered = self.spectral_filter.filter(x_patches, idx)
                
                if return_spectral_info:
                    spectral_info[idx]['after'] = x_filtered.detach().clone()
                
                x = torch.cat([x[:, :self.start_index, :], x_filtered], dim=1)
                current_batch_spectral_loss += self.spectral_filter.filters[str(idx)].get_regularization_loss()

            if use_reg and idx in self.reg_layer_indices:
                attention_map = layer.attn.attn_map
                attention_weights.append(attention_map[:, :, 1:, 1:])  # Remove CLS token

            if idx == max(self.feature_layer_indices):  
                break

        # 多层融合（即插即用）：启用时通过 fusion 模块融合，关闭时直接使用最后一层
        fusion_weights = None
        if self.use_multiscale_fusion and self.fusion is not None:
            if return_fusion_weights:
                features, fusion_weights = self.fusion(out, return_weights=True)
            else:
                features = self.fusion(out)
        else:
            features = out[0]  # 单层特征（已是最后/最深层）

        if use_reg:
            return features, attention_weights, current_batch_spectral_loss, spectral_info, fusion_weights
        else:
            return features, None, current_batch_spectral_loss, spectral_info, fusion_weights
    
    def get_spectral_reg_loss(self):
        """
        获取频域过滤器的正则化损失
        
        Returns:
            reg_loss: 正则化损失，如果没有频域过滤器则返回 0.0
        """
        if self.spectral_filter is not None:
            return self.spectral_filter.get_total_regularization_loss()
        return 0.0
                
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.layer_norm_1(x + self.dropout1(attn_output))
        x = x + self.dropout2(self.linear(x))

        return x

class Discriminator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_patches, num_layers=1, num_heads=12, dropout_rate=0):
        super(Discriminator, self).__init__()
        self.transformer_encoder = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout_rate) for _ in range(num_layers)])
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )
        self.positional_encodings = nn.Parameter(torch.randn(num_patches, embed_dim))

    def forward(self, x):
        x = x + self.positional_encodings.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = self.output_layer(x).squeeze(-1)

        return x

class PatchGuard(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        # Get fusion params with defaults for backward compatibility
        fusion_type = getattr(args, 'fusion_type', 'gated_attention')
        fusion_dim = getattr(args, 'fusion_dim', None)
        
        # Get spectral filtering params with defaults
        spectral_filter_layers = getattr(args, 'spectral_filter_layers', None)
        spectral_gate_reduction = getattr(args, 'spectral_gate_reduction', 4)
        spectral_reg_type = getattr(args, 'spectral_reg_type', 'l1')
        spectral_reg_weight = getattr(args, 'spectral_reg_weight', 0.01)
        spectral_strength = getattr(args, 'spectral_strength', 0.1)
        
        # Get plug-and-play module toggles with defaults
        use_multiscale_fusion = getattr(args, 'use_multiscale_fusion', True)
        use_spectral_filter = getattr(args, 'use_spectral_filter', True)
        
        self.feature_extractor = FeatureExtractor(
            args.hf_path, args.feature_layers, args.reg_layers, 
            args.image_size, device, fusion_type, fusion_dim,
            spectral_filter_layers=spectral_filter_layers,
            spectral_gate_reduction=spectral_gate_reduction,
            spectral_reg_type=spectral_reg_type,
            spectral_reg_weight=spectral_reg_weight,
            spectral_strength=spectral_strength,
            use_multiscale_fusion=use_multiscale_fusion,
            use_spectral_filter=use_spectral_filter
        )

        embed_dim = self.feature_extractor.embed_dim
        self.num_patches = self.feature_extractor.num_patches
        self.patch_size = self.feature_extractor.patch_size
        self.patches_per_side = int(np.sqrt(self.num_patches))

        self.discriminator = Discriminator(embed_dim, args.hidden_dim, self.num_patches, args.dsc_layers, args.dsc_heads, 0.2).to(device)

    def forward(self, x, bypass_spectral=False):
        embeddings, _, _, _, _ = self.feature_extractor(x, False, bypass_spectral=bypass_spectral)
        scores = self.discriminator(embeddings)
        return scores

    def forward_with_details(self, x):
        """前向传播并返回所有中间结果，用于可视化"""
        embeddings, attn_weights, spectral_loss, spectral_info, fusion_weights = \
            self.feature_extractor(x, use_reg=False, return_spectral_info=True, return_fusion_weights=True)
        scores = self.discriminator(embeddings)
        return scores, spectral_info, fusion_weights