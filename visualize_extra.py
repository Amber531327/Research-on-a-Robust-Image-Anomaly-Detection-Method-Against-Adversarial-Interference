"""
PatchGuard 扩展可视化模块

包含 3 个独立的可视化函数：
1. 伪异常生成可视化：原始图 / 伪异常图 / 伪异常掩码
2. 频域门控过滤可视化：干净幅度谱 / 受攻击幅度谱 / 门控抑制后幅度谱
3. 多尺度融合空间门控权重可视化：各层融合权重热力图
"""

import os
import math
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹出窗口

from patchguard import PatchGuard
from pseudo_anomaly import AnomalyGenerator
from attack import pgd_attack
from utils import get_dataloader, load_model, patchify, label_patch


# ============================================================
# 1. 伪异常生成可视化
# ============================================================
def visualize_pseudo_anomaly(args, num_samples=16):
    """
    可视化 Foreground-Aware 伪异常生成过程。
    输出：原始图、伪异常图、伪异常掩码，分文件夹存放。
    """
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # 使用训练集（伪异常生成依赖训练数据和前景掩码）
    train_loader, _ = get_dataloader(
        args.image_size, args.dataset_dir, args.dataset, args.class_name,
        args.train_batch_size, args.test_batch_size, args.num_workers, args.seed
    )

    anomaly_generator = AnomalyGenerator(args.dataset, args.class_name, args.seed)

    # 创建输出目录
    base_dir = Path("./vis_output/pseudo_anomaly")
    (base_dir / "original").mkdir(exist_ok=True, parents=True)
    (base_dir / "augmented").mkdir(exist_ok=True, parents=True)
    (base_dir / "mask").mkdir(exist_ok=True, parents=True)

    count = 0
    for batch in train_loader:
        images = batch[0]              # [B, 3, H, W]
        foreground_masks = batch[3]    # [B, 1, H, W] or [B, H, W]

        augmented_images, augmented_masks = anomaly_generator(images, foreground_masks)

        for b in range(images.shape[0]):
            if count >= num_samples:
                break

            # 原始图像
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(images[b].permute(1, 2, 0).numpy())
            ax.set_title("Original", fontsize=12)
            ax.axis('off')
            plt.savefig(base_dir / "original" / f"img{count}.png", bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

            # 伪异常图像
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(augmented_images[b].permute(1, 2, 0).numpy())
            ax.set_title("Pseudo Anomaly", fontsize=12)
            ax.axis('off')
            plt.savefig(base_dir / "augmented" / f"img{count}.png", bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

            # 伪异常掩码
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            mask_np = augmented_masks[b].squeeze().numpy()
            ax.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            ax.set_title("Pseudo Anomaly Mask", fontsize=12)
            ax.axis('off')
            plt.savefig(base_dir / "mask" / f"img{count}.png", bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

            count += 1

        if count >= num_samples:
            break

    print(f"[Pseudo Anomaly Vis] Done, saved {count} sets of images to {base_dir}")


# ============================================================
# 2. 频域门控过滤可视化
# ============================================================
def _normalize_for_display(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if float(max_val - min_val) < 1e-8:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)


def visualize_adv_noise(args, num_samples=16, noise_scale=10.0):
    """
    Visualize adversarial perturbation with three panels:
    A) clean image x
    B) adversarial image x + delta
    C) amplified perturbation delta = x_adv - x
    """
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloader(
        args.image_size, args.dataset_dir, args.dataset, args.class_name,
        args.train_batch_size, args.test_batch_size, args.num_workers, args.seed
    )

    model = PatchGuard(args, device)
    if args.checkpoint_dir.endswith('.pth'):
        model_path = args.checkpoint_dir
    else:
        model_path = os.path.join(args.checkpoint_dir, f"patchguard_{args.dataset}_{args.class_name}.pth")
    load_model(model, model_path)
    model.eval()

    base_dir = Path("./vis_output/adv_noise")
    base_dir.mkdir(exist_ok=True, parents=True)

    # Keep attack parameters consistent with adversarial training configuration.
    epsilon = getattr(args, "epsilon_train", 8 / 255)
    steps = getattr(args, "step_train", 10)
    noise_scale = getattr(args, "noise_vis_scale", noise_scale)
    num_samples = getattr(args, "noise_vis_num_samples", num_samples)

    count = 0
    for images, _, masks, _ in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        patch_masks = label_patch(patchify(masks, model.patch_size)).to(device)

        with torch.set_grad_enabled(True):
            adv_images = pgd_attack(model, images.clone(), patch_masks, epsilon, steps)

        noises = adv_images - images

        for b in range(images.shape[0]):
            if count >= num_samples:
                break

            clean_np = images[b].detach().cpu().permute(1, 2, 0).numpy()
            adv_np = adv_images[b].detach().cpu().permute(1, 2, 0).numpy()

            noise = noises[b].detach().cpu()
            noise_amplified = noise * noise_scale
            noise_vis = _normalize_for_display(noise_amplified + 0.5)
            noise_np = noise_vis.permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(np.clip(clean_np, 0, 1))
            axes[0].set_title("A. Clean Image (x)", fontsize=12, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(np.clip(adv_np, 0, 1))
            axes[1].set_title("B. Adversarial Image (x+delta)", fontsize=12, fontweight='bold')
            axes[1].axis('off')

            axes[2].imshow(np.clip(noise_np, 0, 1))
            axes[2].set_title(f"C. Amplified Noise ({noise_scale:.1f}x)", fontsize=12, fontweight='bold')
            axes[2].axis('off')

            plt.suptitle(f"PGD params from training: eps={epsilon:.6f}, steps={steps}", fontsize=12)
            plt.tight_layout()
            plt.savefig(base_dir / f"adv_noise_{count}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            count += 1

        if count >= num_samples:
            break

    print(f"[Adv Noise Vis] Done, saved {count} images to {base_dir}")


def _compute_amplitude_spectrum(features, spatial_size):
    """
    将 patch 特征转换为 2D 幅度谱。
    
    Args:
        features: [B, N, D] patch 特征
        spatial_size: int, 空间边长 (H = W = sqrt(N))
    
    Returns:
        amplitude: [B, H, W//2+1] 幅度谱（通道维度已取均值）
    """
    B, N, D = features.shape
    H = W = spatial_size
    # [B, N, D] -> [B, D, H, W]
    feat_2d = features.permute(0, 2, 1).reshape(B, D, H, W)
    # FFT
    freq = torch.fft.rfft2(feat_2d, norm='ortho')
    # 幅度
    magnitude = torch.abs(freq)  # [B, D, H, W//2+1]
    # 沿通道维度取均值
    magnitude_mean = magnitude.mean(dim=1)  # [B, H, W//2+1]
    return magnitude_mean


def visualize_spectral_filter(args, num_samples=8):
    """
    可视化频域门控过滤模块的作用。
    对每张图展示：干净特征幅度谱 / 受攻击特征幅度谱 / 门控抑制后幅度谱
    """
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloader(
        args.image_size, args.dataset_dir, args.dataset, args.class_name,
        args.train_batch_size, args.test_batch_size, args.num_workers, args.seed
    )

    model = PatchGuard(args, device)
    if args.checkpoint_dir.endswith('.pth'):
        model_path = args.checkpoint_dir
    else:
        model_path = os.path.join(args.checkpoint_dir, f"patchguard_{args.dataset}_{args.class_name}.pth")
    load_model(model, model_path)
    model.eval()

    # 确定要可视化的频域过滤层
    spectral_layers = getattr(args, 'spectral_filter_layers', [3, 6])
    if not spectral_layers:
        print("[Spectral Filter Vis] Error: spectral_filter_layers not configured, cannot visualize.")
        return

    vis_layer = spectral_layers[0]  # 默认可视化第一个频域过滤层
    patches_per_side = model.patches_per_side

    base_dir = Path(f"./vis_output/spectral_filter/layer_{vis_layer}")
    base_dir.mkdir(exist_ok=True, parents=True)

    count = 0
    with torch.no_grad():
        for images, _, masks, _ in test_loader:
            images, masks = images.to(device), masks.to(device)

            # --- 干净图像的频域特征 ---
            _, clean_spectral_info, _ = model.forward_with_details(images)

            if vis_layer not in clean_spectral_info:
                print(f"[Spectral Filter Vis] Warning: Layer {vis_layer} not found in spectral filter layers, skipping.")
                return

            clean_before = clean_spectral_info[vis_layer]['before']  # [B, N, D]
            clean_after = clean_spectral_info[vis_layer]['after']    # [B, N, D]

            # --- 对抗攻击图像的频域特征 ---
            with torch.set_grad_enabled(True):
                adv_images = pgd_attack(
                    model, images.clone(),
                    label_patch(patchify(masks, model.patch_size)),
                    getattr(args, 'epsilon_visualization', 8/255),
                    getattr(args, 'step_visualization', 10)
                )

            with torch.no_grad():
                _, adv_spectral_info, _ = model.forward_with_details(adv_images)
                adv_before = adv_spectral_info[vis_layer]['before']  # [B, N, D]
                adv_after = adv_spectral_info[vis_layer]['after']    # [B, N, D]

            # 计算幅度谱
            clean_amp = _compute_amplitude_spectrum(clean_before, patches_per_side)
            adv_amp = _compute_amplitude_spectrum(adv_before, patches_per_side)
            gated_amp = _compute_amplitude_spectrum(adv_after, patches_per_side)

            for b in range(images.shape[0]):
                if count >= num_samples:
                    break

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                # 原始图像
                axes[0].imshow(images[b].cpu().permute(1, 2, 0).numpy())
                axes[0].set_title("Input Image", fontsize=13, fontweight='bold')
                axes[0].axis('off')

                # 获取三个频谱的 numpy 数据 (log scale)
                clean_spec = torch.log1p(clean_amp[b]).cpu().numpy()
                adv_spec = torch.log1p(adv_amp[b]).cpu().numpy()
                gated_spec = torch.log1p(gated_amp[b]).cpu().numpy()

                # ==========================================
                # 【排查打印】：在终端输出绝对差异的最大值
                # ==========================================
                max_diff = np.max(np.abs(adv_spec - gated_spec))
                print(f"[Debug] Sample {count} - Max absolute diff between attacked and gated spectrum: {max_diff:.6f}")
                
                if max_diff == 0:
                    print("Warning: Difference is 0! The module had no effect.")

                # ==========================================
                # 【统一色阶】：找到三张图的全局最大和最小值
                # ==========================================
                vmin = min(clean_spec.min(), adv_spec.min(), gated_spec.min())
                vmax = max(clean_spec.max(), adv_spec.max(), gated_spec.max())

                # 干净幅度谱
                im1 = axes[1].imshow(clean_spec, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                axes[1].set_title(f"Clean Spectrum (Layer {vis_layer})", fontsize=13, fontweight='bold')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                # 受攻击幅度谱
                im2 = axes[2].imshow(adv_spec, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                axes[2].set_title(f"Attacked Spectrum (Layer {vis_layer})", fontsize=13, fontweight='bold')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                # ==========================================
                # 【神图改造】：将第4张图改为“被切除的噪声残差”
                # ==========================================
                # 计算门控模块到底切掉了什么：|Attacked - Gated|
                diff_spec = np.abs(adv_spec - gated_spec)
                
                # 残差图用 'hot' 或 'magma' 颜色带，黑色代表没变化，亮色代表被切除的噪声
                im3 = axes[3].imshow(diff_spec, cmap='hot', aspect='auto')
                axes[3].set_title(f"Suppressed Noise (Attacked - Gated)", fontsize=13, fontweight='bold')
                axes[3].axis('off')
                plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

                plt.suptitle(f"Spectral Gating Filter Visualization — Layer {vis_layer}", 
                           fontsize=15, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig(base_dir / f"spectral_vis_{count}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)

                count += 1
    print(f"[Spectral Filter Vis] Done, saved {count} images to {base_dir}")


# ============================================================
# 3. 多尺度融合空间门控权重可视化
# ============================================================
def visualize_fusion_weights(args, num_samples=20):
    """
    可视化多尺度融合模块中每个 Patch 的层级融合权重。
    选取最浅层和最深层的权重绘制热力图对比。
    """
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloader(
        args.image_size, args.dataset_dir, args.dataset, args.class_name,
        args.train_batch_size, args.test_batch_size, args.num_workers, args.seed
    )

    model = PatchGuard(args, device)
    if args.checkpoint_dir.endswith('.pth'):
        model_path = args.checkpoint_dir
    else:
        model_path = os.path.join(args.checkpoint_dir, f"patchguard_{args.dataset}_{args.class_name}.pth")
    load_model(model, model_path)
    model.eval()

    # 检查是否启用了多尺度融合
    if not getattr(args, 'use_multiscale_fusion', True):
        print("[Fusion Weight Vis] Error: Multi-scale fusion not enabled, cannot visualize.")
        return

    feature_layers = args.feature_layers  # e.g., [3, 6, 9, 12]
    num_scales = len(feature_layers)
    patches_per_side = model.patches_per_side

    base_dir = Path("./vis_output/fusion_weights")
    base_dir.mkdir(exist_ok=True, parents=True)

    count = 0
    with torch.no_grad():
        for images, _, masks, _ in test_loader:
            images = images.to(device)

            _, _, fusion_weights = model.forward_with_details(images)

            if fusion_weights is None:
                print("[Fusion Weight Vis] Error: fusion_weights is None, check if fusion_type supports returning weights.")
                return

            # fusion_weights: [B, N, num_scales]
            for b in range(images.shape[0]):
                if count >= num_samples:
                    break

                # 创建子图：原始图像 + 每层权重热力图
                fig, axes = plt.subplots(1, num_scales + 1, figsize=(5 * (num_scales + 1), 5))

                # 原始图像
                axes[0].imshow(images[b].cpu().permute(1, 2, 0).numpy())
                axes[0].set_title("Input Image", fontsize=13, fontweight='bold')
                axes[0].axis('off')

                # 各层权重热力图
                for s in range(num_scales):
                    weight_map = fusion_weights[b, :, s].cpu().numpy()  # [N]
                    weight_map_2d = weight_map.reshape(patches_per_side, patches_per_side)

                    im = axes[s + 1].imshow(weight_map_2d, cmap='hot', vmin=0, vmax=1,
                                            interpolation='bilinear')
                    axes[s + 1].set_title(f"Layer {feature_layers[s]} Weight", 
                                         fontsize=13, fontweight='bold')
                    axes[s + 1].axis('off')
                    plt.colorbar(im, ax=axes[s + 1], fraction=0.046, pad=0.04)

                plt.suptitle("Multi-Scale Fusion Spatial Gating Weights", 
                           fontsize=15, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig(base_dir / f"fusion_weights_{count}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)

                count += 1

            if count >= num_samples:
                break

    print(f"[Fusion Weight Vis] Done, saved {count} images to {base_dir}")
