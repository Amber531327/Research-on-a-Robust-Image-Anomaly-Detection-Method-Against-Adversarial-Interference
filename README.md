# PatchGuard++: 面向对抗干扰的鲁棒图像异常检测方法研究

## 项目简介

本项目在 PatchGuard 异常检测框架的基础上，针对对抗样本攻击导致检测性能急剧下降的问题，提出了两项即插即用的改进模块：

1. **多尺度特征融合 (Multi-Scale Feature Fusion)**：通过空间门控注意力机制融合 ViT 不同层级的特征，使模型同时利用浅层细粒度信息与深层语义信息进行异常判别。
2. **频域门控过滤 (Spectral Gating Filter)**：在 ViT 浅层对特征进行 FFT → 可学习门控 → IFFT 的频域变换，自适应抑制对抗噪声主导的高频分量，保护下游层的特征质量。

两项模块均设计为**即插即用 (Plug-and-Play)**，可通过命令行开关独立启用或关闭，无需修改核心网络结构。

## 项目声明

| 项目信息 | 内容 |
|---------|------|
| 项目名称 | 面向对抗干扰的鲁棒图像异常检测方法研究 |
| 英文名称 | Research on a Robust Image Anomaly Detection Method Against Adversarial Interference |
| 作者 | 唐奂 |
| 作者单位 | 暨南大学网络空间安全学院 |
| 开发语言 | Python 3.10+ |
| 深度学习框架 | PyTorch 2.10 |
| 骨干网络 | DINOv2 ViT-S/14 (via `timm`) |
| 对抗攻击方法 | PGD (Projected Gradient Descent) |
| 核心技术 | 多尺度特征融合、频域门控过滤、对抗训练、伪异常生成 |

## 项目目录结构

```
├── main.py                  # 主入口，命令行参数解析与模式调度
├── patchguard.py            # 模型核心：PatchGuard、FeatureExtractor、Discriminator、
│                            #   MultiScaleFeatureFusion 融合模块
├── spectral_filter.py       # 频域门控过滤模块 (SpectralGatingModule、
│                            #   MultiLayerSpectralFilter)
├── train_model.py           # 训练流程：训练循环、分组学习率、checkpoint 保存
├── test_model.py            # 测试流程：多类别多模型批量测试、结果汇总
├── attack.py                # PGD 对抗攻击生成
├── pseudo_anomaly.py        # 前景感知伪异常生成器 (AnomalyGenerator)
├── loss.py                  # 损失函数：BCE + 注意力正则化 (KL散度)
├── dataset.py               # 数据集类：MVTec、VisA、MPDD、BTAD、DTD、
│                            #   BraTS2021、HeadCT、WFDD
├── utils.py                 # 工具函数：数据加载、patchify、AUC 计算、模型保存/加载
├── download_data.py         # 自动下载 MVTec AD / VisA 数据集及前景掩码
├── download_weight.py       # 自动下载预训练模型权重
├── visualize.py             # 异常热力图可视化（干净样本 & 对抗样本对比）
├── visualize_extra.py       # 扩展可视化：伪异常生成、频域过滤效果、融合权重、对抗噪声
├── visualize_loss.py        # 训练/验证损失曲线绘制
├── test_multiscale.py       # 多尺度融合模块单元测试
├── test_spectral.py         # 频域过滤模块单元测试
├── patchguard_notebook.ipynb  # Jupyter Notebook 交互式版本
├── requirements.txt         # 依赖包列表
└── plots/                   # 可视化输出目录
```

## 方法概览

### 基线模型：PatchGuard

PatchGuard 是一种基于 patch 级别的图像异常检测方法：

- 使用 DINOv2 ViT-S/14 作为特征提取器，提取多层 patch 特征
- 引入 Attention Regularization 防止 ViT 的 attention 塌缩
- 使用前景感知的伪异常生成策略产生训练数据
- 通过 Transformer Discriminator 对每个 patch 进行正常/异常二分类
- 支持对抗训练以提升模型鲁棒性

### 改进模块一：多尺度特征融合

从 ViT 的多个中间层提取特征，通过空间门控注意力 (Gated Attention) 逐 patch 学习各层融合权重，生成融合后的统一特征表示。支持四种融合策略：

- `concat`：简单通道拼接
- `concat_proj`：拼接后线性投影
- `scalar_attention`：全局可学习标量权重
- `gated_attention`：逐 patch 空间门控权重（推荐）

### 改进模块二：频域门控过滤

在 ViT 浅层（如 Layer 3、Layer 6）对隐状态特征进行频域过滤：

1. 将 patch 序列还原为 2D 空间结构
2. 通过 RFFT2 变换到频域，分离幅度与相位
3. 门控网络基于幅度学习保留/抑制权重
4. 通过 IRFFT2 变换回空间域
5. 残差连接保护空间细节信息

通过 `spectral_strength` 参数控制全局过滤强度，`spectral_adv_only` 使频域过滤器仅从对抗样本中学习。

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.6（推荐）
- PyTorch 2.10+

### 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：

| 包名 | 用途 |
|------|------|
| torch, torchvision | 深度学习框架 |
| timm | DINOv2 ViT 模型加载 |
| transformers, huggingface_hub | HuggingFace 模型仓库 |
| albumentations, kornia | 数据增强 |
| scikit-learn | AUC 评估指标 |
| matplotlib, rich | 可视化与终端美化 |

### 下载数据集

```bash
# 下载 MVTec AD 数据集
python download_data.py --dataset mvtec

# 下载 VisA 数据集
python download_data.py --dataset visa
```

### 下载预训练权重

```bash
# 下载 MVTec 数据集 leather 类别的预训练权重
python download_weight.py --dataset mvtec --class_name leather --checkpoint_dir ./checkpoints/

# 也可以通过主脚本下载
python main.py --mode test --dataset mvtec --class_name leather \
    --dataset_dir ./datasets/MVTec --use_weight_prep \
    --checkpoint_dir ./checkpoints/
```

### 训练

```bash
# 基础训练（不使用改进模块）
python main.py --mode train --dataset mvtec --class_name leather \
    --dataset_dir ./datasets/MVTec --epochs 300 --checkpoint_dir ./checkpoints/

# 使用全部改进模块训练
python main.py --mode train --dataset mvtec --class_name leather \
    --dataset_dir ./datasets/MVTec --epochs 300 \
    --feature_layers 3 6 9 12 --reg_layers 3 6 9 12 \
    --fusion_type gated_attention --fusion_dim 384 \
    --spectral_filter_layers 3 6 --spectral_strength 0.1 \
    --checkpoint_dir ./checkpoints/
```

### 测试

```bash
# 单模型测试（干净 + 对抗样本）
python main.py --mode test --dataset mvtec --class_name leather \
    --dataset_dir ./datasets/MVTec \
    --feature_layers 3 6 9 12 --reg_layers 3 6 9 12 \
    --fusion_type gated_attention --fusion_dim 384 \
    --spectral_filter_layers 3 6 \
    --epsilon_test 8 --step_test 1000 \
    --checkpoint_dir ./checkpoints/
```

### 可视化

```bash
# 异常热力图可视化（干净 vs 对抗）
python main.py --mode visualization --dataset mvtec --class_name leather \
    --dataset_dir ./datasets/MVTec \
    --feature_layers 3 6 9 12 --reg_layers 3 6 9 12 \
    --fusion_type gated_attention --fusion_dim 384 \
    --checkpoint_dir ./checkpoints/

# 伪异常生成可视化
python main.py --mode vis_pseudo --dataset mvtec --class_name leather \
    --dataset_dir ./datasets/MVTec

# 对抗噪声可视化
python main.py --mode vis_adv_noise --dataset mvtec --class_name leather \
    --dataset_dir ./datasets/MVTec \
    --feature_layers 3 6 9 12 --reg_layers 3 6 9 12 \
    --checkpoint_dir ./checkpoints/

# 损失曲线绘制
python main.py --mode vis_loss --dataset mvtec --class_name leather
```

## 主要命令行参数

### 必需参数

| 参数 | 说明 |
|------|------|
| `--mode` | 运行模式：`train` / `test` / `visualization` / `vis_pseudo` / `vis_spectral` / `vis_fusion` / `vis_loss` / `vis_adv_noise` |
| `--class_name` | 数据集类别名称（支持多类别） |
| `--dataset` | 数据集名称：`mvtec` / `visa` / `mpdd` / `btad` / `dtd` / `brats2021` / `headct` / `wfdd` |
| `--dataset_dir` | 数据集根目录 |

### 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--feature_layers` | `3 6 9 12` | 提取特征的 ViT 层索引 |
| `--reg_layers` | `3 6 9 12` | 施加注意力正则化的层索引 |
| `--fusion_type` | `gated_attention` | 多尺度融合策略 |
| `--fusion_dim` | `384` | 融合后特征维度 |
| `--spectral_filter_layers` | `3 6` | 应用频域过滤的层索引 |
| `--spectral_strength` | `0.3` | 频域过滤强度 (0.0=不过滤, 1.0=全过滤) |
| `--spectral_reg_type` | `l1` | 频域门控正则化类型 |
| `--spectral_reg_weight` | `0.01` | 频域正则化权重 |
| `--spectral_adv_only` | `False` | 仅从对抗样本学习频域过滤（保护洁净样本性能） |

### 即插即用模块开关

| 参数 | 说明 |
|------|------|
| `--no_multiscale_fusion` | 关闭多尺度特征融合 |
| `--no_spectral_filter` | 关闭频域过滤模块 |

### 对抗训练/测试

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--adv_train` / `--no_adv_train` | 启用 | 是否进行对抗训练 |
| `--adv_test` / `--no_adv_test` | 启用 | 是否进行对抗测试 |
| `--epsilon_train` | `8` | 训练对抗扰动强度 (像素值, 0-255) |
| `--epsilon_test` | `8` | 测试对抗扰动强度 |
| `--step_train` | `10` | 训练 PGD 迭代步数 |
| `--step_test` | `1000` | 测试 PGD 迭代步数 |

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | `1000` | 训练轮数 |
| `--train_batch_size` | `16` | 训练批次大小 |
| `--lr` | `0.0008` | 学习率 |
| `--lr_decay_factor` | `0.0125` | 余弦退火最小学习率比例 |
| `--image_size` | `224` | 输入图像尺寸 |
| `--seed` | `0` | 随机种子 |

## 评估指标

模型性能使用以下指标评估：

- **Image-level AUC**：图像级异常检测的 ROC 曲线下面积，反映模型区分正常/异常图像的能力
- **Pixel-level AUC**：像素级异常定位的 ROC 曲线下面积，反映模型对异常区域的精确定位能力

在干净样本和 PGD 对抗攻击样本上分别评估这两项指标。


## 单元测试

```bash
# 测试多尺度融合模块（4种配置组合）
python test_multiscale.py

# 测试频域过滤模块
python test_spectral.py
```

## License

本项目仅用于学术研究与毕业设计，不提供商业使用许可。

---

**Author**: 唐奂  
**Institution**: 暨南大学网络空间安全学院  
**Date**: 2026
