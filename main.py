import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse

from train_model import run_train
from test_model import run_test
from visualize import visualize_heatmap
from visualize_extra import visualize_pseudo_anomaly, visualize_spectral_filter, visualize_fusion_weights, visualize_adv_noise
from visualize_loss import visualize_loss_curves
from download_data import download_dataset
from download_weight import download_weights

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, default="train", choices=["train", "test", "visualization", "vis_pseudo", "vis_spectral", "vis_fusion", "vis_loss", "vis_adv_noise"])
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class_name", type=str, nargs='*', required=True, default=[], help="One or more class names to train/test")
    parser.add_argument("--dataset", type=str, required=True, default="mvtec")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--lr_decay_factor", type=float, default=0.0125)
    parser.add_argument("--lr_adaptor", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0.00001)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_tqdm", action="store_false", dest='use_tqdm', default=True)

    # feature extractor config
    parser.add_argument("--hf_path", type=str, default='vit_small_patch14_dinov2.lvd142m')
    parser.add_argument("--feature_layers", type=int, nargs='+', default=[3,6,9,12], help="Layers to extract features (shallow, mid, deep).")
    parser.add_argument("--reg_layers", type=int, nargs='+', default=[3,6,9,12], help="Layers to apply attention regularization (should match feature_layers).")
    parser.add_argument("--fusion_type", type=str, default='gated_attention', 
                        choices=['concat', 'concat_proj', 'scalar_attention', 'gated_attention'],
                        help="Multi-scale fusion strategy: 'concat' (simple), 'concat_proj' (with projection), 'scalar_attention' (learnable weights), 'gated_attention' (spatial gating)")
    parser.add_argument("--fusion_dim", type=int, default=384, help="Output dimension for fusion (only used with concat_proj/attention, default: base embed_dim)")
    # discriminator config
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dsc_layers", type=int, default=1)
    parser.add_argument("--dsc_heads", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--smoothing_sigma", type=int, default=6)
    parser.add_argument("--smoothing_radius", type=int, default=7)
    # adversarial attack config
    parser.add_argument("--attack_type", type=str, default="PGD")
    parser.add_argument("--no_adv_train", action="store_false", dest='adv_train', default=True)
    parser.add_argument("--no_adv_test", action="store_false", dest='adv_test', default=True)
    parser.add_argument("--epsilon_train", type=float, default=8)
    parser.add_argument("--epsilon_test", type=float, nargs='+', default=[8])
    parser.add_argument("--epsilon_visualization", type=float, default=8)
    parser.add_argument("--step_train", type=int, default=10)
    parser.add_argument("--step_test", type=int, default=1000)
    parser.add_argument("--step_visualization", type=int, default=10)
    parser.add_argument("--noise_vis_scale", type=float, default=10.0,
                        help="Amplification factor for adversarial noise visualization.")
    parser.add_argument("--noise_vis_num_samples", type=int, default=16,
                        help="Number of samples to visualize in adversarial noise mode.")
    # regularizer config
    parser.add_argument("--no_reg", action="store_false", dest='use_reg', default=True)
    parser.add_argument("--reg_type", type=str, default="KL_divergence", choices=["KL_divergence"])

    # spectral filtering config (频域过滤配置)
    parser.add_argument("--spectral_filter_layers", type=int, nargs='*', default=[3,6],
                        help="Layers to apply spectral filtering (e.g., 3 6). Default: None (disabled)")
    parser.add_argument("--spectral_gate_reduction", type=int, default=4,
                        help="Channel reduction ratio for spectral gate network")
    parser.add_argument("--spectral_reg_type", type=str, default='l1', 
                        choices=['l1', 'l2', 'none'],
                        help="Regularization type for spectral gates")
    parser.add_argument("--spectral_reg_weight", type=float, default=0.01,
                        help="Weight for spectral gate regularization loss")
    parser.add_argument("--spectral_strength", type=float, default=0.3,
                        help="Spectral filter blending strength (0.0=bypass, 1.0=full filter). Controls max filter intensity.")

    # plug-and-play module toggles (即插即用模块开关)
    parser.add_argument("--no_multiscale_fusion", action="store_false", dest='use_multiscale_fusion', default=True,
                        help="Disable multi-scale feature fusion (use last feature layer only)")
    parser.add_argument("--no_spectral_filter", action="store_false", dest='use_spectral_filter', default=True,
                        help="Disable spectral frequency filtering module")
    parser.add_argument("--spectral_adv_only", action="store_true", default=False,
                        help="Only train spectral filter on adversarial samples (freeze on clean samples to prevent clean degradation)")

    # prepare data and weight
    parser.add_argument("--use_data_prep", action="store_true", default=False)
    parser.add_argument("--use_weight_prep", action="store_true", default=False)


    args = parser.parse_args()
    return args

def main(args):
    if args.use_data_prep:
        download_dataset(args.dataset)
    if args.use_weight_prep:
        download_weights(args.dataset, args.class_name, args.checkpoint_dir)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    elif args.mode == "visualization":
        class_names = args.class_name if isinstance(args.class_name, list) else [args.class_name]
        for cn in class_names:
            args.class_name = cn
            visualize_heatmap(args)
    elif args.mode == "vis_pseudo":
        # 伪异常生成可视化（使用训练集，不需要 checkpoint）
        class_names = args.class_name if isinstance(args.class_name, list) else [args.class_name]
        for cn in class_names:
            args.class_name = cn
            visualize_pseudo_anomaly(args)
    elif args.mode == "vis_spectral":
        # 频域门控过滤可视化（需要 checkpoint）
        class_names = args.class_name if isinstance(args.class_name, list) else [args.class_name]
        for cn in class_names:
            args.class_name = cn
            visualize_spectral_filter(args)
    elif args.mode == "vis_fusion":
        # 多尺度融合权重可视化（需要 checkpoint）
        class_names = args.class_name if isinstance(args.class_name, list) else [args.class_name]
        for cn in class_names:
            args.class_name = cn
            visualize_fusion_weights(args)
    elif args.mode == "vis_adv_noise":
        class_names = args.class_name if isinstance(args.class_name, list) else [args.class_name]
        for cn in class_names:
            args.class_name = cn
            visualize_adv_noise(args)
    elif args.mode == "vis_loss":
        visualize_loss_curves(args)

if __name__ == "__main__":
    args = parse_args()
    args.epsilon_train /= 255
    args.epsilon_visualization /= 255
    args.epsilon_test = [epsilon / 255 for epsilon in args.epsilon_test]
    main(args)
