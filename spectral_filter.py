"""
频域门控过滤模块 (Spectral Gating Filter Module)

用于在ViT浅层（Layer 3, 6）对特征进行频域过滤，
去除高频对抗噪声，提高模型对对抗样本的鲁棒性。

流程: FFT -> 可学习门控 -> IFFT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGatingModule(nn.Module):
    """
    频域门控模块：FFT -> 可学习门控 -> IFFT
    
    将ViT的patch序列还原为2D结构，使用rfft2进行变换，
    基于幅度(Magnitude)进行门控，保留相位(Phase)信息。
    """
    
    def __init__(self, embed_dim, num_patches, gate_reduction=4, 
                 regularization='l1', reg_weight=0.01, spectral_strength=0.1):
        """
        Args:
            embed_dim: 特征维度 (D)
            num_patches: patch数量 (N = H * W)
            gate_reduction: 门控网络通道缩减比例
            regularization: 正则化类型 'l1', 'l2', 或 'none'
            reg_weight: 正则化权重
            spectral_strength: 全局过滤强度 (0.0=完全不过滤, 1.0=完全过滤)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.spatial_size = int(math.sqrt(num_patches))  # H = W = sqrt(N)
        self.regularization = regularization
        self.reg_weight = reg_weight
        self.spectral_strength = spectral_strength
        
        # 频域的宽度（rfft2 输出宽度为 W//2 + 1）
        self.freq_w = self.spatial_size // 2 + 1
        
        # 门控网络：基于幅度生成每个频率分量的权重
        # 输入: [B, D, H, W//2+1] 幅度谱
        # 输出: [B, D, H, W//2+1] 门控权重
        hidden_dim = embed_dim // gate_reduction
        
        self.gate_net = nn.Sequential(
            # 第一层卷积：通道压缩
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 第二层卷积：空间处理（3x3卷积捕获局部频率模式）
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 第三层卷积：恢复通道数并生成门控权重
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid()  # 输出范围 [0, 1]，作为每个频率分量的保留权重
        )
        
        # 可学习的残差混合系数（初始化为 0.0 → sigmoid(0)=0.5，偏向保留原始信息）
        self.blend_alpha = nn.Parameter(torch.tensor(0.0))
        
        # 用于累积正则化损失
        self._reg_loss = 0.0
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化门控网络权重"""
        for m in self.gate_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        last_conv = self.gate_net[-2]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.xavier_normal_(last_conv.weight, gain=0.01) # 权重极小
            nn.init.constant_(last_conv.bias, 5.0)              # 偏置极大
    
    def forward(self, x):
        """
        前向传播（带残差混合）
        
        Args:
            x: [B, N, D] 输入特征序列
            
        Returns:
            output: [B, N, D] 残差混合后的特征
        """
        B, N, D = x.shape
        H = W = self.spatial_size
        
        assert N == H * W, f"Expected {H*W} patches, got {N}"
        assert D == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {D}"
        
        # Step 1: 将序列 reshape 为 2D 空间结构
        # [B, N, D] -> [B, D, H, W]
        x_2d = x.permute(0, 2, 1).reshape(B, D, H, W)
        
        # Step 2: 快速傅里叶变换 (FFT)
        # 使用 rfft2 (实数 FFT)，输出形状: [B, D, H, W//2+1] (复数)
        x_freq = torch.fft.rfft2(x_2d, norm='ortho')
        
        # Step 3: 分离幅度和相位
        magnitude = torch.abs(x_freq)  # [B, D, H, W//2+1]
        phase = torch.angle(x_freq)    # [B, D, H, W//2+1]
        
        # Step 4: 使用门控网络生成权重
        # 基于幅度信息学习哪些频率成分应该被保留
        gate_weights = self.gate_net(magnitude)  # [B, D, H, W//2+1], 值在 [0, 1]
        
        # Step 5: 计算正则化损失（鼓励门控接近 pass-through）
        self._compute_reg_loss(gate_weights)
        
        # Step 6: 应用门控 - 仅调整幅度，保留相位
        filtered_magnitude = magnitude * gate_weights
        
        # Step 7: 重建频域信号（极坐标形式转回复数）
        x_freq_filtered = filtered_magnitude * torch.exp(1j * phase)
        
        # Step 8: 傅里叶逆变换 (IFFT)
        x_filtered_2d = torch.fft.irfft2(x_freq_filtered, s=(H, W), norm='ortho')
        
        # Step 9: 残差混合 — 保护空间细节
        # blend_alpha 通过 sigmoid 映射到 [0,1]，再乘以 spectral_strength 控制上限
        alpha = torch.sigmoid(self.blend_alpha) * self.spectral_strength
        x_blended_2d = alpha * x_filtered_2d + (1 - alpha) * x_2d
        
        # Step 10: Reshape 回序列形式
        # [B, D, H, W] -> [B, N, D]
        output = x_blended_2d.reshape(B, D, N).permute(0, 2, 1)
        
        return output
    
    def _compute_reg_loss(self, gate_weights):
        """
        计算门控权重的正则化损失（pass-through 方向）
        
        鼓励 gate_weights 接近 1.0（保留原始频率信息），
        只在模型学习到确实需要抑制的频率时才偏离 1.0。
        
        Args:
            gate_weights: [B, D, H, W//2+1] 门控权重，范围 [0, 1]
        """
        if self.regularization == 'l1':
            # L1 正则化：惩罚 gate 偏离 1.0 的程度（鼓励 pass-through）
            self._reg_loss = self.reg_weight * torch.mean(torch.abs(1.0 - gate_weights))
        elif self.regularization == 'l2':
            # L2 正则化：惩罚 gate 偏离 1.0 的平方（鼓励 pass-through）
            self._reg_loss = self.reg_weight * torch.mean((1.0 - gate_weights) ** 2)
        else:
            self._reg_loss = 0.0
    
    def get_regularization_loss(self):
        """
        获取当前累积的正则化损失
        
        Returns:
            reg_loss: 标量正则化损失
        """
        return self._reg_loss
    
    def reset_regularization_loss(self):
        """重置正则化损失（在每个训练步骤开始时调用）"""
        self._reg_loss = 0.0


class MultiLayerSpectralFilter(nn.Module):
    """
    多层频域过滤器管理类
    
    管理多个层的频域过滤模块，提供统一的接口获取总正则化损失。
    """
    
    def __init__(self, layer_indices, embed_dim, num_patches, 
                 gate_reduction=4, regularization='l1', reg_weight=0.01, 
                 spectral_strength=0.1, device='cuda'):
        """
        Args:
            layer_indices: 需要应用频域过滤的层索引列表，如 [3, 6]
            embed_dim: 特征维度
            num_patches: patch 数量
            gate_reduction: 门控网络通道缩减比例
            regularization: 正则化类型
            reg_weight: 正则化权重
            spectral_strength: 全局过滤强度 (0.0=不过滤, 1.0=完全过滤)
            device: 设备
        """
        super().__init__()
        
        self.layer_indices = layer_indices
        self.filters = nn.ModuleDict()
        
        for idx in layer_indices:
            self.filters[str(idx)] = SpectralGatingModule(
                embed_dim=embed_dim,
                num_patches=num_patches,
                gate_reduction=gate_reduction,
                regularization=regularization,
                reg_weight=reg_weight,
                spectral_strength=spectral_strength
            ).to(device)
    
    def filter(self, x, layer_idx):
        """
        对指定层的特征应用频域过滤
        
        Args:
            x: [B, N, D] 特征
            layer_idx: 层索引
            
        Returns:
            filtered_x: [B, N, D] 过滤后的特征
        """
        key = str(layer_idx)
        if key in self.filters:
            return self.filters[key](x)
        return x
    
    def get_total_regularization_loss(self):
        """获取所有过滤器的总正则化损失"""
        total_loss = 0.0
        for filter_module in self.filters.values():
            loss = filter_module.get_regularization_loss()
            if isinstance(loss, torch.Tensor):
                total_loss = total_loss + loss
            else:
                total_loss = total_loss + loss
        return total_loss
    
    def reset_all_regularization_losses(self):
        """重置所有过滤器的正则化损失"""
        for filter_module in self.filters.values():
            filter_module.reset_regularization_loss()
    
    def __contains__(self, layer_idx):
        """检查某层是否有频域过滤器"""
        return str(layer_idx) in self.filters
