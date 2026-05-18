
## 使用PatchGuard原模型训练
(graduation) D:\HuDie\AI\test\test>python main.py --mode test --dataset mvtec --class_name leather --dataset_dir ./datasets/MVTec --step_test 1000 
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()

================================================================================
Testing model 1/6: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_80.pth
================================================================================

Detected class name: leather
Model loaded from D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_80.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9076 │
│ Pixel-level AUC │ 0.8696 │
└─────────────────┴────────┘
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6107 │
│ Pixel-level AUC │ 0.5845 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_80.pth
================================================================================


================================================================================
Testing model 2/6: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_120.pth
================================================================================

Detected class name: leather
Model loaded from D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_120.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8838 │
│ Pixel-level AUC │ 0.8959 │
└─────────────────┴────────┘
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7687 │
│ Pixel-level AUC │ 0.5859 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_120.pth
================================================================================


================================================================================
Testing model 3/6: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_180.pth
================================================================================

Detected class name: leather
Model loaded from D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_180.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9528 │
│ Pixel-level AUC │ 0.9449 │
└─────────────────┴────────┘
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.3376 │
│ Pixel-level AUC │ 0.0434 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_180.pth
================================================================================


================================================================================
Testing model 4/6: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_240.pth
================================================================================

Detected class name: leather
Model loaded from D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_240.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9592 │
│ Pixel-level AUC │ 0.9350 │
└─────────────────┴────────┘
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.1688 │
│ Pixel-level AUC │ 0.0898 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_240.pth
================================================================================


================================================================================
Testing model 5/6: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_280.pth
================================================================================

Detected class name: leather
Model loaded from D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_280.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9450 │
│ Pixel-level AUC │ 0.9445 │
└─────────────────┴────────┘
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.1080 │
│ Pixel-level AUC │ 0.0769 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_280.pth
================================================================================


================================================================================
Testing model 6/6: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_mvtec_carpet_pgd10_last_epoch.pth
================================================================================

Detected class name: leather
Warning: Checkpoint file not found: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_mvtec_carpet_pgd10_last_epoch.pth
Skipping this model...


(graduation) D:\HuDie\AI\test\test>python main.py --mode test --dataset mvtec --class_name leather --dataset_dir ./datasets/MVTec --step_test 1000 
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()

================================================================================
Testing model 1/3: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_100.pth
================================================================================

Detected class name: leather
Model loaded from D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_100.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9100 │
│ Pixel-level AUC │ 0.8612 │
└─────────────────┴────────┘
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.5333 │
│ Pixel-level AUC │ 0.5257 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_100.pth
================================================================================


================================================================================
Testing model 2/3: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_140.pth
================================================================================

Detected class name: leather
Model loaded from D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_140.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7272 │
│ Pixel-level AUC │ 0.8843 │
└─────────────────┴────────┘
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6430 │
│ Pixel-level AUC │ 0.5678 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_140.pth
================================================================================


================================================================================
Testing model 3/3: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_160.pth
================================================================================

Detected class name: leather
Model loaded from D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_160.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8845 │
│ Pixel-level AUC │ 0.9127 │
└─────────────────┴────────┘
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
D:\ProgramData\Miniconda\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7065 │
│ Pixel-level AUC │ 0.2190 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: D:\HuDie\AI\test\test\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_160.pth
================================================================================

## 没有进行以下改进后结果，但单独测试频域过滤模块不达预期
无残差连接（主因）— 过滤器完全替换了 ViT 隐状态，FFT→门控→IFFT 后丢失了像素级定位需要的细粒度空间信息
L1 正则化方向反了 — 当前 L1 惩罚 mean(|gate|) 鼓励 gate→0（更多抑制），应该反过来惩罚 mean(|1-gate|) 鼓励 gate→1（保留更多信息，只在必要时抑制）
缺少全局强度控制 — in-stream 过滤在 layer 3/6 直接改写隐状态，缺少一个全局旋钮来软化过滤强度
让频域过滤器 只从对抗样本中学习

(graduation) PS E:\Graduation_Project\Improvedlast_PatchGuard> python main.py --mode test --dataset mvtec --dataset_dir ./datasets/MVTec --class_name leather --feature_layers 3 6 9 12 --reg_layers 3 6 9 12 --fusion_type gated_attention --fusion_dim 384 --spectral_filter_layers 3 6 --epsilon_test 8 --step_test 1000

================================================================================
Testing model 1/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_60.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_60.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8947 │
│ Pixel-level AUC │ 0.8869 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7089 │
│ Pixel-level AUC │ 0.6267 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_60.pth
================================================================================


================================================================================
Testing model 2/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_80.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_80.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7527 │
│ Pixel-level AUC │ 0.8912 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.5635 │
│ Pixel-level AUC │ 0.6199 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_80.pth
================================================================================


================================================================================
Testing model 3/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_100.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_100.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8998 │
│ Pixel-level AUC │ 0.9078 │
└─────────────────┴────────┘
D:\anaconda3\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7531 │
│ Pixel-level AUC │ 0.6575 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_100.pth
================================================================================


================================================================================
Testing model 4/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_120.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_120.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\anaconda3\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9620 │
│ Pixel-level AUC │ 0.8995 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6016 │
│ Pixel-level AUC │ 0.3383 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_120.pth
================================================================================


================================================================================
Testing model 5/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_140.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_140.pth
Dataloaders for dataset mvtec and class leather have been prepared.
D:\anaconda3\envs\graduation\lib\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:1000: The handshake operation timed out>
  data = fetch_version_info()
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9674 │
│ Pixel-level AUC │ 0.9347 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.2418 │
│ Pixel-level AUC │ 0.2112 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_140.pth
================================================================================


================================================================================
Testing model 6/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_160.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_160.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9056 │
│ Pixel-level AUC │ 0.8961 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.4959 │
│ Pixel-level AUC │ 0.4981 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_160.pth
================================================================================


================================================================================
Testing model 7/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_180.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_180.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7877 │
│ Pixel-level AUC │ 0.8796 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6821 │
│ Pixel-level AUC │ 0.5874 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_180.pth
================================================================================


================================================================================
Testing model 8/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_200.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_200.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9005 │
│ Pixel-level AUC │ 0.8793 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.5808 │
│ Pixel-level AUC │ 0.5813 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_200.pth
================================================================================


================================================================================
Testing model 9/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_220.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_220.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9385 │
│ Pixel-level AUC │ 0.9033 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7171 │
│ Pixel-level AUC │ 0.5804 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_220.pth
================================================================================


================================================================================
Testing model 10/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_240.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_240.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9096 │
│ Pixel-level AUC │ 0.8996 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7014 │
│ Pixel-level AUC │ 0.5948 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_240.pth
================================================================================


================================================================================
Testing model 11/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_260.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_260.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8998 │
│ Pixel-level AUC │ 0.9016 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7728 │
│ Pixel-level AUC │ 0.5411 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_260.pth
================================================================================


================================================================================
Testing model 12/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_280.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_280.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8886 │
│ Pixel-level AUC │ 0.8890 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6807 │
│ Pixel-level AUC │ 0.5062 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_280.pth
================================================================================


================================================================================
Testing model 13/13: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_mvtec_leather_pgd10_last_epoch.pth
================================================================================

Detected class name: leather
Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_mvtec_leather_pgd10_last_epoch.pth
Dataloaders for dataset mvtec and class leather have been prepared.
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.9069 │
│ Pixel-level AUC │ 0.9009 │
└─────────────────┴────────┘
         PGD attack         
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6624 │
│ Pixel-level AUC │ 0.5182 │
└─────────────────┴────────┘

================================================================================
Completed testing for model: E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_mvtec_leather_pgd10_last_epoch.pth
================================================================================

#进行上述所有改进并先提取干净特征给 fusion，再让频域过滤修改 x 供后续层传播结果
(graduation) PS E:\Graduation_Project\Improvedlast_PatchGuard> python main.py --mode test --dataset mvtec --dataset_dir ./datasets/MVTec --class_name leather --epochs 300 --feature_layers 3 6 9 12 --reg_layers 3 6 9 12 --fusion_type gated_attention --fusion_dim 384 --spectral_filter_layers 3 6 --spectral_reg_type l1 --spectral_reg_weight 0.01 --spectral_strength 0.1 --spectral_adv_only   

################################################################################
# Starting multi-category multi-model testing
# Total categories: 1
# Categories: ['leather']
################################################################################


================================================================================
= CATEGORY 1/1: leather
= Number of models to test: 14
================================================================================

Dataloaders for dataset mvtec and class leather have been prepared.

------------------------------------------------------------
[leather] Testing model 1/14
Checkpoint: patchguard_epoch_40.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_40.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8190 │
│ Pixel-level AUC │ 0.8027 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.5659 │
│ Pixel-level AUC │ 0.7270 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 2/14
Checkpoint: patchguard_epoch_60.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_60.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6912 │
│ Pixel-level AUC │ 0.8142 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.2456 │
│ Pixel-level AUC │ 0.6832 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 3/14
Checkpoint: patchguard_epoch_80.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_80.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7928 │
│ Pixel-level AUC │ 0.8030 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7429 │
│ Pixel-level AUC │ 0.7652 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 4/14
Checkpoint: patchguard_epoch_100.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_100.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7150 │
│ Pixel-level AUC │ 0.8295 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.4436 │
│ Pixel-level AUC │ 0.6315 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 5/14
Checkpoint: patchguard_epoch_120.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_120.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7255 │
│ Pixel-level AUC │ 0.8443 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6399 │
│ Pixel-level AUC │ 0.6105 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 6/14
Checkpoint: patchguard_epoch_140.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_140.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8200 │
│ Pixel-level AUC │ 0.8553 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6179 │
│ Pixel-level AUC │ 0.5533 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 7/14
Checkpoint: patchguard_epoch_160.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_160.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6970 │
│ Pixel-level AUC │ 0.8070 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.3118 │
│ Pixel-level AUC │ 0.6819 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 8/14
Checkpoint: patchguard_epoch_180.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_180.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8434 │
│ Pixel-level AUC │ 0.8657 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6199 │
│ Pixel-level AUC │ 0.5412 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 9/14
Checkpoint: patchguard_epoch_200.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_200.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8760 │
│ Pixel-level AUC │ 0.8825 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.6053 │
│ Pixel-level AUC │ 0.4762 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 10/14
Checkpoint: patchguard_epoch_220.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_220.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8356 │
│ Pixel-level AUC │ 0.8609 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.7415 │
│ Pixel-level AUC │ 0.6103 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 11/14
Checkpoint: patchguard_epoch_240.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_240.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8665 │
│ Pixel-level AUC │ 0.9020 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.4983 │
│ Pixel-level AUC │ 0.3067 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 12/14
Checkpoint: patchguard_epoch_260.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_260.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8743 │
│ Pixel-level AUC │ 0.9189 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.2965 │
│ Pixel-level AUC │ 0.0693 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 13/14
Checkpoint: patchguard_epoch_280.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_epoch_280.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8329 │
│ Pixel-level AUC │ 0.9192 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.3675 │
│ Pixel-level AUC │ 0.0285 │
└─────────────────┴────────┘

------------------------------------------------------------
[leather] Testing model 14/14
Checkpoint: patchguard_mvtec_leather_pgd10_last_epoch.pth
------------------------------------------------------------

Model loaded from E:\Graduation_Project\Improvedlast_PatchGuard\checkpoints_mvtec_leather_pgd10_epochs300\patchguard_mvtec_leather_pgd10_last_epoch.pth
      Clean Performance     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.8295 │
│ Pixel-level AUC │ 0.9226 │
└─────────────────┴────────┘
         PGD attack
 (eps=0.03137254901960784,  
         step=1000)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     Metric      ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Image-level AUC │ 0.3227 │
│ Pixel-level AUC │ 0.0263 │
└─────────────────┴────────┘

================================================================================
= Completed testing for category: leather
================================================================================
