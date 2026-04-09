# FaceMe2: Dual-Branch MVP for Blind Face Restoration

本项目是基于 AAAI 2025 论文 **FaceMe** 的简化与改进版本（SDSC 8016 课程项目）。

## 🔗 原版相关链接 (Original References)
- **原始代码仓库**: [modyu-liu/FaceMe](https://github.com/modyu-liu/FaceMe)
- **原始论文**: [FaceMe: Robust Blind Face Restoration with Personal Identification](https://arxiv.org/abs/2501.05177)

---

## 🎯 项目现状与目标 (Project Overview & Status)
原始的 FaceMe 使用了 Diffusion 模型和 ControlNet 等较为复杂的结构。为了提升训练与推理效率，同时满足资源限制要求，本项目将其核心思想提取并重构为一个 **双分支 MVP (Dual-Branch MVP)** 架构。

目前，项目已经成功跑通了基于该新架构的核心训练代码，修复了混合精度训练 (FP16/FP32) 的显存报错，并支持了自定义训练集规模等功能。

### 🧩 运行流水线与核心架构 (Running Pipeline)
模型主要由两个并行的分支构成：
1. **质量分支 (Quality Branch)**:
   - **核心模型**: SwinIR
   - **功能**: 负责从严重退化的低质量 (LQ) 输入图像中恢复出高质量的高频细节和纹理。
   - **损失函数**: 采用 L1 Loss + Perceptual Loss (感知损失) 来确保生成图像的整体视觉质量。
2. **身份分支 (Identity Branch)**:
   - **核心模型**: ResNet-18 (作为身份特征编码器)
   - **功能**: 提取并约束生成图像与原始高质量图像之间的面部身份特征一致性。
   - **损失函数**: 采用 ArcFace Loss (Identity Loss) 防止生成人脸出现“不像本人”的问题。
3. **训练策略**:
   - 整体采用 **3 阶段交替训练策略 (3-phase alternating strategy)**，在不同的阶段动态调整各个分支的损失权重，最终达到图像重建质量与身份保真度的最佳平衡。

---

## 📊 数据集 (Dataset)
- **基础数据**: 训练主要基于 **CelebA-HQ** 或 **FFHQ** 等高质量人脸数据集。
- **动态退化 (Dynamic Degradation)**: 在数据加载时 (`dataset.py`)，我们通过实时加入随机模糊 (Blur)、降采样 (Downsample)、噪声 (Noise) 和 JPEG 压缩等手段，动态生成对应的低质量 (LQ) 输入。
- **身份特征预处理**: 在训练之前，需预先提取图像的 ID Embedding 和 CLIP Embedding (以 `.npy` 格式保存)，通过 `train.json` 进行管理和读取。

---

## 🚀 运行步骤 (How to Run)

### 1. 环境准备
本项目使用了 `accelerate` 进行多卡分布式训练。
```bash
# 激活预配置的 conda 环境
conda activate wei310
```

### 2. 数据准备与预处理
如果还没有 `train.json`，需要通过脚本提取特征并生成：
```bash
python utils/create_train_json.py \
    --ffhq_dir 'data/train/FFHQ/' \
    --ffhq_emb_dir 'output/id_emb/' \
    --ffhqref_emb_dir 'output/clip_emb/' \
    --save_dir 'output/train_json/'
```

### 3. 启动训练 (Training)
我们提供了一个封装好的多卡启动脚本 `run_train.sh`，内部已经配置好了环境变量和优化器策略。

```bash
# 后台启动训练，并将日志输出到 train_log.txt
bash run_train.sh
```

**`run_train.sh` 核心参数说明:**
- `--pretrained_model_name_or_path`: 预训练模型的路径 (如 RealVisXL_V3.0)。
- `--train_data_dir`: 上一步生成的 `train.json` 路径。
- `--resolution`: 图像输入分辨率 (例如 128 或 512)。
- `--mixed_precision`: 支持 `fp16` 混合精度训练，以降低显存占用。
- `--max_train_samples`: **(调试/测试用)** 可以传入一个整数限制训练集规模。例如传入 `1055` (总数据集 10555 的 10%)，可以快速验证模型。

### 4. 查看日志
训练日志可以通过 `tail` 或 `cat` 实时查看：
```bash
tail -f train_log.txt
```
模型 Checkpoints 将默认保存在 `./output/train_results/` 中。
