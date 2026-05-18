import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from patchguard import PatchGuard
from utils import get_dataloader, save_model, log_loss, patchify, label_patch
from loss import Loss
from attack import pgd_attack
from pseudo_anomaly import AnomalyGenerator
from visualize_loss import plot_loss_curve
import os 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _set_spectral_filter_grad(model, requires_grad):
    """设置频域过滤器参数的梯度开关"""
    sf = model.feature_extractor.spectral_filter
    if sf is not None:
        sf.requires_grad_(requires_grad)

def train_step(model, anomaly_generator, train_loader, optimizer, criterion, use_reg, device, args):
    total_sample = 0
    total_loss = 0

    spectral_adv_only = getattr(args, 'spectral_adv_only', False)

    batch_iterator = tqdm(train_loader, disable=not args.use_tqdm, desc="Training Batches")
    for batch in batch_iterator:
        loss = 0
        accumulated_spectral_loss = 0.0  # 累积所有前向传播的频域正则化损失

        normal_data = [batch[0].to(device)]

        if args.adv_train:
            images = batch[0].to(device).clone()
            adv_normal_images = pgd_attack(model, images, torch.zeros(images.shape[0], model.num_patches).to(device), args.epsilon_train, args.step_train, bypass_spectral=False)
            normal_data.append(adv_normal_images)

        # ---- 处理正常数据（洁净 + 对抗正常图）----
        for i, imgs in enumerate(normal_data):
            is_clean = (i == 0)  # 第一个是洁净图像

            # 洁净样本冻结频域过滤器（可选）
            if spectral_adv_only and is_clean:
                _set_spectral_filter_grad(model, False)

            features, attn_weights, spectral_loss, _, _ = model.feature_extractor(imgs, use_reg)

            # 洁净样本的 spectral_loss 不参与反向传播
            if spectral_adv_only and is_clean:
                spectral_loss = spectral_loss.detach() if isinstance(spectral_loss, torch.Tensor) else spectral_loss
            accumulated_spectral_loss += spectral_loss

            # 解冻频域过滤器
            if spectral_adv_only and is_clean:
                _set_spectral_filter_grad(model, True)

            scores_true = model.discriminator(features)

            masks_true = torch.zeros(features.shape[0], features.shape[1]).to(device)
            loss += criterion(scores_true, masks_true, attn_weights)

        images = batch[0].clone()
        foreground_masks = batch[3]

        augmented_images, augmented_masks = anomaly_generator(images, foreground_masks)
        augmented_masks = label_patch(patchify(augmented_masks, model.patch_size))
        augmented_images, augmented_masks = augmented_images.to(device), augmented_masks.to(device)

        anomaly_data = [augmented_images]
        if args.adv_train:
            adv_distorted_images = pgd_attack(model, augmented_images.clone(), augmented_masks, args.epsilon_train, args.step_train, bypass_spectral=True)
            anomaly_data.append(adv_distorted_images)

        # ---- 处理异常数据（洁净 + 对抗异常图）----
        for i, imgs in enumerate(anomaly_data):
            is_clean = (i == 0)  # 第一个是洁净异常图

            # 洁净样本冻结频域过滤器（可选）
            if spectral_adv_only and is_clean:
                _set_spectral_filter_grad(model, False)

            features, attn_weights, spectral_loss, _, _ = model.feature_extractor(imgs, use_reg)

            if spectral_adv_only and is_clean:
                spectral_loss = spectral_loss.detach() if isinstance(spectral_loss, torch.Tensor) else spectral_loss
            accumulated_spectral_loss += spectral_loss

            if spectral_adv_only and is_clean:
                _set_spectral_filter_grad(model, True)
            
            scores_aug = model.discriminator(features)

            loss += criterion(scores_aug, augmented_masks, attn_weights)

        optimizer.zero_grad()
        
        # 将累积的频域正则化损失加入总损失
        combined_loss = loss + accumulated_spectral_loss
        
        combined_loss.backward()
        optimizer.step()

        total_sample += images.shape[0]
        total_loss += loss.item() * images.shape[0]
    
    return total_loss / total_sample


def validate_step(model, val_loader, criterion, use_reg, device):
    model.eval()
    total_sample = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device)
            masks = batch[2].to(device)
            patch_masks = label_patch(patchify(masks, model.patch_size)).to(device)

            features, attn_weights, _, _, _ = model.feature_extractor(images, use_reg)
            scores = model.discriminator(features)
            loss = criterion(scores, patch_masks, attn_weights)

            batch_size = images.shape[0]
            total_sample += batch_size
            total_loss += loss.item() * batch_size

    model.train()
    if total_sample == 0:
        return 0.0
    return total_loss / total_sample


def train(model, anomaly_generator, train_loader, val_loader, optimizer, lr_scheduler, criterion, use_reg, epochs, device, ckpt_path, args, class_name):
    model.train()
    
    epoch_iterator = tqdm(range(epochs), disable=not args.use_tqdm, desc="Epochs")
    for epoch in epoch_iterator:
        train_loss = train_step(model, anomaly_generator, train_loader, optimizer, criterion, use_reg, device, args)
        val_loss = validate_step(model, val_loader, criterion, use_reg, device)
        lr_scheduler.step()

        epoch_iterator.set_postfix(train_loss=train_loss, val_loss=val_loss)
        train_log_file_path = f"{args.dataset}_{class_name}_loss_log"
        val_log_file_path = f"{args.dataset}_{class_name}_val_loss_log"
        log_loss(epoch, train_loss, train_log_file_path)
        log_loss(epoch, val_loss, val_log_file_path)

        if (epoch % 20 == 0) and (epoch > 20):
            save_model(model, ckpt_path + f"/patchguard_epoch_{epoch}.pth")


def run_train(args):
    set_seed(args.seed)
    
    # Support multiple class names
    class_names = args.class_name if isinstance(args.class_name, list) else [args.class_name]
    
    for idx, class_name in enumerate(class_names):
        print(f"\n{'='*80}")
        print(f"Training class {idx+1}/{len(class_names)}: {class_name}")
        print(f"{'='*80}\n")
        
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoints_{args.dataset}_{class_name}_pgd{args.step_train}_epochs{args.epochs}")
        os.makedirs(ckpt_path, exist_ok=True)

        device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
        model = PatchGuard(args, device).to(device)

        train_loader, val_loader = get_dataloader(args.image_size, args.dataset_dir, args.dataset, class_name, args.train_batch_size, args.test_batch_size, args.num_workers, args.seed)
        
        # 分组学习率：预训练 backbone 最低 lr，新增模块正常 lr
        backbone_params = list(model.feature_extractor.pretrained_model.parameters())
        discriminator_params = list(model.discriminator.parameters())
        param_groups = [
            {'params': backbone_params, 'lr': args.lr * 0.1},       # 预训练模型微调
            {'params': discriminator_params, 'lr': args.lr * 0.5},   # 鉴别器适中
        ]
        if model.feature_extractor.fusion is not None:
            param_groups.append({'params': list(model.feature_extractor.fusion.parameters()), 'lr': args.lr,'weight_decay': 1e-2})
        if model.feature_extractor.spectral_filter is not None:
            param_groups.append({'params': list(model.feature_extractor.spectral_filter.parameters()), 'lr': args.lr,'weight_decay': 1e-2})
        optimizer = optim.AdamW(param_groups)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * args.lr_decay_factor)
        
        criterion = Loss(args.reg_type, device, model.num_patches)

        anomaly_generator = AnomalyGenerator(args.dataset, class_name, args.seed)  

        train_log_file_path = f"{args.dataset}_{class_name}_loss_log"
        val_log_file_path = f"{args.dataset}_{class_name}_val_loss_log"
        if os.path.exists(train_log_file_path):
            os.remove(train_log_file_path)
        if os.path.exists(val_log_file_path):
            os.remove(val_log_file_path)

        train(model, anomaly_generator, train_loader, val_loader, optimizer, lr_scheduler, criterion, args.use_reg, args.epochs, device, ckpt_path, args, class_name)

        save_model(model, os.path.join(ckpt_path, f"patchguard_{args.dataset}_{class_name}_pgd{args.step_train}_last_epoch.pth"))
        plot_loss_curve(train_log_file_path, args.dataset, class_name, val_log_path=val_log_file_path)
        
        print(f"\n{'='*80}")
        print(f"Completed training for class: {class_name}")
        print(f"{'='*80}\n")
        
        # Clean up GPU memory
        del model, train_loader, val_loader, optimizer, lr_scheduler, criterion, anomaly_generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
