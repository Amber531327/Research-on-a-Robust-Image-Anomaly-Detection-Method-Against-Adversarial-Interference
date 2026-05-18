import torch
import re
import os

from patchguard import PatchGuard
from utils import get_dataloader, load_model, patchify, label_patch, get_auc, display_results
from attack import pgd_attack

import glob

# ===== 配置区域：在此指定需要测试的通用路径和类别 =====

# 通用基础目录路径
BASE_CHECKPOINT_DIR = r"/content/drive/MyDrive/Improvedlast_PatchGuard327"

# 类别文件夹命名规则的通配符模板，自动用类别名替换 {category}
# 例：匹配 "checkpoints_mvtec_bottle_pgd10_epochs300" 文件夹ke
CHECKPOINT_DIR_PATTERN = "checkpoints_mvtec_{category}_*"

# 需要测试的类别列表
TEST_CATEGORIES = [
    "tile",
]

def get_category_checkpoints():
    checkpoints = {}
    for category in TEST_CATEGORIES:
        # 构建并匹配类别对应的文件夹路径
        dir_pattern = os.path.join(BASE_CHECKPOINT_DIR, CHECKPOINT_DIR_PATTERN.format(category=category))
        matched_dirs = glob.glob(dir_pattern)
        
        if not matched_dirs:
            print(f"Warning: Checkpoint directory not found for category: {category} (Pattern: {dir_pattern})")
            continue
            
        # 默认取第一个匹配的文件夹
        category_dir = matched_dirs[0] 
        
        # 查找该文件夹下的所有 .pth 模型文件
        pth_pattern = os.path.join(category_dir, "*.pth")
        pth_files = glob.glob(pth_pattern)
        
        if pth_files:
            # 排序规则：按 epoch 数字从小到大排序，带有 last_epoch 的文件排在最后
            def sort_key(filepath):
                filename = os.path.basename(filepath)
                if "last_epoch" in filename:
                    return float('inf')
                match = re.search(r'epoch_(\d+)', filename)
                if match:
                    return int(match.group(1))
                return -1
                
            pth_files.sort(key=sort_key)
            checkpoints[category] = pth_files
        else:
            print(f"Warning: No .pth files found in {category_dir}")
            
    return checkpoints



def test(model, test_loader, device, args, adv_test, epsilon=8 / 255, steps=10):
    model.eval()

    test_scores = []
    test_labels = []
    test_masks = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            if adv_test:
                masks = label_patch(patchify(batch[2], model.patch_size)).to(device)
                with torch.set_grad_enabled(True):
                    if args.attack_type == "PGD":
                        images = pgd_attack(model, images, masks, epsilon, steps)

            masks = batch[2].to(device)
            scores = model(images)

            test_scores.append(scores.cpu())
            test_labels.append(labels.cpu())
            test_masks.append(masks.cpu())

    image_auc, pixel_auc = get_auc(test_scores, test_labels, test_masks, model.patches_per_side, args.smoothing_sigma,
                                   args.smoothing_radius, args.top_k)

    return image_auc, pixel_auc


def run_test(args):
    CATEGORY_CHECKPOINTS = get_category_checkpoints()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # 如果CATEGORY_CHECKPOINTS字典为空，使用原有的单模型测试逻辑
    if not CATEGORY_CHECKPOINTS:
        print("Warning: CATEGORY_CHECKPOINTS is empty. Falling back to single model testing.")
        class_name = args.class_name[0] if isinstance(args.class_name, list) else args.class_name
        model = PatchGuard(args, device).to(device)
        load_model(model, args.checkpoint_dir + f"patchguard_{args.dataset}_{class_name}.pth")
        _, test_loader = get_dataloader(args.image_size, args.dataset_dir, args.dataset, class_name,
                                        args.train_batch_size, args.test_batch_size, args.num_workers, args.seed)

        image_auc, pixel_auc = test(model, test_loader, device, args, False)
        display_results({"Image-level AUC": image_auc, "Pixel-level AUC": pixel_auc}, "Clean Performance")

        if args.adv_test:
            epsilons = args.epsilon_test
            step = args.step_test

            for epsilon in epsilons:
                image_auc, pixel_auc = test(model, test_loader, device, args, True, epsilon, step)
                display_results({"Image-level AUC": image_auc, "Pixel-level AUC": pixel_auc},
                                f"{args.attack_type} attack (eps={epsilon}, step={step})")
        return

    # 存储所有测试结果的汇总
    all_results = {}
    total_categories = len(CATEGORY_CHECKPOINTS)
    
    print(f"\n{'#' * 80}")
    print(f"# Starting multi-category multi-model testing")
    print(f"# Total categories: {total_categories}")
    print(f"# Categories: {list(CATEGORY_CHECKPOINTS.keys())}")
    print(f"{'#' * 80}\n")

    # 遍历每个类别
    for cat_idx, (class_name, checkpoint_paths) in enumerate(CATEGORY_CHECKPOINTS.items()):
        print(f"\n{'=' * 80}")
        print(f"= CATEGORY {cat_idx + 1}/{total_categories}: {class_name}")
        print(f"= Number of models to test: {len(checkpoint_paths)}")
        print(f"{'=' * 80}\n")
        
        # 初始化该类别的结果存储
        all_results[class_name] = {
            "clean": [],
            "adversarial": {}
        }
        
        # 为该类别加载一次数据集（所有模型共用）
        _, test_loader = get_dataloader(args.image_size, args.dataset_dir, args.dataset, class_name,
                                        args.train_batch_size, args.test_batch_size, args.num_workers, args.seed)
        
        # 遍历该类别的所有模型
        for model_idx, checkpoint_path in enumerate(checkpoint_paths):
            print(f"\n{'-' * 60}")
            print(f"[{class_name}] Testing model {model_idx + 1}/{len(checkpoint_paths)}")
            print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
            print(f"{'-' * 60}\n")

            # 加载模型
            model = PatchGuard(args, device).to(device)
            
            # 支持相对路径和绝对路径
            full_checkpoint_path = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(args.checkpoint_dir, checkpoint_path)

            if not os.path.exists(full_checkpoint_path):
                print(f"Warning: Checkpoint file not found: {full_checkpoint_path}")
                print("Skipping this model...\n")
                del model
                continue

            load_model(model, full_checkpoint_path)

            # 清洁数据测试
            image_auc, pixel_auc = test(model, test_loader, device, args, False)
            display_results({"Image-level AUC": image_auc, "Pixel-level AUC": pixel_auc},
                            f"[{class_name}] Clean Performance")
            
            # 存储结果
            all_results[class_name]["clean"].append({
                "checkpoint": os.path.basename(checkpoint_path),
                "image_auc": image_auc,
                "pixel_auc": pixel_auc
            })

            # 对抗攻击测试
            if args.adv_test:
                epsilons = args.epsilon_test
                step = args.step_test

                for epsilon in epsilons:
                    eps_key = f"eps_{epsilon}"
                    if eps_key not in all_results[class_name]["adversarial"]:
                        all_results[class_name]["adversarial"][eps_key] = []
                    
                    image_auc, pixel_auc = test(model, test_loader, device, args, True, epsilon, step)
                    display_results({"Image-level AUC": image_auc, "Pixel-level AUC": pixel_auc},
                                    f"[{class_name}] {args.attack_type} attack (eps={epsilon}, step={step})")
                    
                    all_results[class_name]["adversarial"][eps_key].append({
                        "checkpoint": os.path.basename(checkpoint_path),
                        "image_auc": image_auc,
                        "pixel_auc": pixel_auc
                    })

            # 清理模型内存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 清理数据加载器
        del test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n{'=' * 80}")
        print(f"= Completed testing for category: {class_name}")
        print(f"{'=' * 80}\n")
    
    # 打印汇总结果
    print_summary(all_results, args)


def print_summary(all_results, args):
    """打印所有类别的测试结果汇总"""
    print(f"\n{'#' * 80}")
    print(f"# SUMMARY OF ALL TEST RESULTS")
    print(f"{'#' * 80}\n")
    
    for class_name, results in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"Category: {class_name}")
        print(f"{'=' * 60}")
        
        # 清洁数据结果
        print("\n--- Clean Performance ---")
        print(f"{'Checkpoint':<50} {'Image AUC':<12} {'Pixel AUC':<12}")
        print("-" * 74)
        for result in results["clean"]:
            print(f"{result['checkpoint']:<50} {result['image_auc']:<12.4f} {result['pixel_auc']:<12.4f}")
        
        # 计算平均值
        if results["clean"]:
            avg_image = sum(r['image_auc'] for r in results['clean']) / len(results['clean'])
            avg_pixel = sum(r['pixel_auc'] for r in results['clean']) / len(results['clean'])
            print("-" * 74)
            print(f"{'AVERAGE':<50} {avg_image:<12.4f} {avg_pixel:<12.4f}")
        
        # 对抗攻击结果
        if args.adv_test and results["adversarial"]:
            for eps_key, adv_results in results["adversarial"].items():
                epsilon = eps_key.replace("eps_", "")
                print(f"\n--- {args.attack_type} Attack (eps={epsilon}) ---")
                print(f"{'Checkpoint':<50} {'Image AUC':<12} {'Pixel AUC':<12}")
                print("-" * 74)
                for result in adv_results:
                    print(f"{result['checkpoint']:<50} {result['image_auc']:<12.4f} {result['pixel_auc']:<12.4f}")
                
                if adv_results:
                    avg_image = sum(r['image_auc'] for r in adv_results) / len(adv_results)
                    avg_pixel = sum(r['pixel_auc'] for r in adv_results) / len(adv_results)
                    print("-" * 74)
                    print(f"{'AVERAGE':<50} {avg_image:<12.4f} {avg_pixel:<12.4f}")


