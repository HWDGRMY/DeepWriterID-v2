# DeepWriterID v2.0：ConvNeXt 架构升级

## 项目简介

本项目是 [DeepWriterID-Replication](https://github.com/HWDGRMY/DeepWriterID-Replication)（v1.0）的架构升级版本。v1.0 使用原始 DCNN 骨干，在 CASIA-OLHWDB 2.0-2.2（1019 位书写者）上达到 **95.09%** 页面级准确率。v2.0 用 **ConvNeXt V2** 替换 DCNN 骨干，在不改动数据流水线的前提下引入现代化架构，为后续伪时序融合与安全检测提供更强的特征提取基础。

> **v1.0 仓库（DCNN 基线、数据准备、硬件约束等）**：https://github.com/HWDGRMY/DeepWriterID-Replication  
> **v2.0 当前状态**：ConvNeXt V2 骨干已实现，正在训练验证

## 版本演进

| 版本 | 核心架构 | 关键成果 | 状态 |
| :--- | :--- | :--- | :--- |
| **v1.0** | 原始 DCNN | 在 CASIA-OLHWDB 2.0-2.2（1019 位书写者）上达到 **95.09%** 页面级准确率 | ✅ 已完成 |
| **v2.0** | ConvNeXt V2 | 目标：骨干升级后准确率不低于 v1.0，并验证伪时序特征的增量价值 | 🚧 开发中 |
| **v2.0 扩展** | 拼接检测 + 生成检测 + 对抗博弈 | 构建红蓝对抗闭环，评估模型在物理篡改与生成式伪造下的鲁棒性 | 📋 规划中 |

> 数据集说明、硬件配置要求、数据划分策略等通用信息，请参阅 [v1.0 仓库 README](https://github.com/HWDGRMY/DeepWriterID-Replication)。

## 🧠 v2.0 架构升级说明

### 为什么升级？

原始 DCNN 架构（2019 年 DeepWriterID 原论文）在当今顶刊审稿中显得缺乏新意。v2.0 用 **ConvNeXt V2** 替换 DCNN，引入现代化骨干网络，提升特征表达能力，并为后续伪时序融合与安全检测提供更强的特征提取基础。

### 升级内容

| 模块 | v1.0 | v2.0 |
| :--- | :--- | :--- |
| **骨干网络** | DCNN（5 层卷积 + 3 层全连接） | ConvNeXt V2-Tiny（适配 64 通道 54×54 输入） |
| **特征维度** | 512 维 | 512 维（对齐 v1.0） |
| **分类头** | Dropout 0.2 + Linear | Dropout 0.2 + Linear |
| **数据流水线** | 路径签名 + DropSegment | 完全复用 v1.0（不变） |
| **训练策略** | 余弦退火 + 断点续训 | 完全复用 v1.0（不变） |

### 为什么选择 ConvNeXt V2？

ConvNeXt V2 是 2023 年 CVPR 论文《ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders》提出的现代化卷积网络。它通过引入大卷积核、LayerNorm、GELU、倒瓶颈结构和 **GRN 层**（Global Response Normalization），将 Transformer 的设计理念融入纯卷积网络，在保持 CNN 高效推理优势的同时，达到与 Swin Transformer 相当的精度。

选择它的理由：
1. **输入尺寸适配**：本任务的特征图为 54×54 小尺寸，不适合 ViT 类架构的全局注意力。
2. **任务特性**：笔迹识别本质上是纹理和几何形状识别，卷积的平移不变性和局部感受野天然适合。
3. **部署友好**：后续需要移植至国产 NPU（麒麟 + 昇腾）平台，卷积算子比自注意力算子更成熟。
4. **消融价值**：可与 Swin Transformer 做对比实验，验证纯卷积架构在细粒度分类任务上的优势。

### 后续规划

- **伪时序双路融合**：从原始轨迹中提取伪速度与曲率序列，用 Transformer Encoder 提取动态特征，与 ConvNeXt V2 静态特征融合。
- **拼接伪造检测**：冻结骨干，挂载 MLP 检测头，识别物理拼接攻击。
- **生成式伪造检测**：训练 CVAE 生成假笔迹，构建对抗博弈框架，评估检测器鲁棒性。
- **国产平台移植**：将模型适配至麒麟 + 昇腾平台。

## 项目目录结构

```text
DeepWriterID-v2/
├── configs/                            # 参数配置文件
│   └── convnext.yaml                   # ConvNeXt V2 超参数
├── data/                               # 数据目录（详见 v1.0 仓库）
│   ├── raw/                            # 存放原始 .wptt 轨迹文件
│   └── features/                       # 存放 metadata.csv
├── src/                                # 核心源码包
│   ├── data/                           # 数据加载与处理
│   │   ├── loader.py                   # 解析 .wptt
│   │   └── dataset.py                  # WPTTDataset (含 DropSegment)
│   ├── models/                         # 模型定义
│   │   └── backbone.py                 # ConvNeXt V2 骨干网络
│   ├── preprocessing/                  # 预处理
│   │   ├── corner.py                   # 拐点检测
│   │   └── segmentation.py             # 伪字符切分
│   ├── features/                       # 特征提取
│   │   └── path_signature.py           # 路径签名
│   ├── augmentation/                   # 数据增强
│   │   └── drop_segment.py             # 核心 DropSegment 逻辑
│   ├── training/                       # 训练引擎
│   │   └── trainer.py                  # 核心训练循环
│   ├── evaluation/                     # 评估引擎
│   │   └── evaluator.py                # 页面级投票准确率计算
│   └── utils/                          # 工具函数
│       └── visualizer.py               # 可视化工具
├── scripts/                            # 可执行入口脚本
│   ├── preprocess.py                   # 预处理 (生成 metadata.csv)
│   ├── train.py                        # 训练入口
│   └── evaluate.py                     # 单独评估模型
├── outputs/                            # 产出物（不上传 GitHub）
│   ├── logs/                           # 训练日志 CSV
│   ├── checkpoints/                    # 模型权重 .pth
│   └── results/                        # 测试可视化图片
├── environment.yml                     # Conda 环境配置
├── setup.py                            # 包安装配置
├── README.md                           # 项目说明
└── LICENSE                             # 开源协议
```

## 快速开始

### 1. 环境配置

建议使用 Conda 创建并激活环境：

```bash
conda env create -f environment.yml
conda activate deepwriterid
```

或使用 pip 安装（确保已安装 PyTorch 与 CUDA 环境）：

```bash
pip install -e .
pip install timm
```

### 2. 数据准备

请按照 [v1.0 仓库的数据准备说明](https://github.com/HWDGRMY/DeepWriterID-Replication) 下载 CASIA-OLHWDB 2.0-2.2 数据集，并运行：

```bash
python scripts/preprocess.py
```

### 3. 训练模型（ConvNeXt V2）

修改 `configs/convnext.yaml` 中的参数，然后运行：

```bash
python scripts/train.py --config configs/convnext.yaml
```

- 训练日志将保存在 `outputs/logs/convnext_training_log.csv`
- 最新模型保存至 `outputs/checkpoints/convnext_latest.pth`
- 刷新记录时额外保存至 `outputs/checkpoints/convnext_best.pth`

**默认配置说明：**

- `epochs：50`
- `batch_size：256`
- `initial_lr：0.0001`（ConvNeXt V2 收敛敏感，学习率略降）
- `weight_decay：0.05`
- `warmup_epochs：5`
- `drop_path：0.1`

训练脚本实现了标准的断点续训功能。训练中断后重新运行，模型权重、优化器动量以及学习率调度器的状态都会被完全恢复。

### 📉 学习率调度策略（余弦退火）

本项目采用 **PyTorch 内置的 `CosineAnnealingLR`** 作为学习率调度器。

- **初始学习率**：`0.0001`
- **总周期（T_max）**：等于设置的 `epochs`（默认 50 轮）
- **变化曲线**：从 `0.0001` 开始，沿平滑余弦曲线在训练结束时下降至接近 `0`
- **核心优势**：相比阶梯式下降，平滑衰减能有效避免训练后期因学习率突变而震荡，帮助权重稳定收敛到最优解

### 4. 评估模型

加载最佳模型并进行完整的页面级测试：

```bash
python scripts/evaluate.py --config configs/convnext.yaml
```

## 📦 模型文件说明（不公开）

本项目的预训练模型文件（`convnext_best.pth`）**不包含在本 GitHub 仓库中**，原因与 v1.0 相同：算力成本高、模型资产需保护。

### 📎 如何获取模型文件？

- **通过 GitHub Issues 提交申请**  
  请访问 [Issues 页面](https://github.com/HWDGRMY/DeepWriterID-v2/issues) 新建 Issue，简要说明使用目的。
- **通过邮件直接联系作者**  
  发送邮件至 **`zhouhao_oss@163.com`**，请附上身份、用途及具体场景。

> **注意**：模型文件仅限申请用途使用，请勿二次分发。

## 📜 第三方代码与引用

本项目在静态分支中使用了以下开源实现，特此致谢并声明引用：

### timm (PyTorch Image Models)

- **仓库**：https://github.com/huggingface/pytorch-image-models
- **许可证**：Apache License 2.0
- **用途**：提供 `convnextv2_tiny` 骨干网络的 PyTorch 实现
- **引用**：
  ```
  @misc{rw2019timm,
    author = {Ross Wightman},
    title = {PyTorch Image Models},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    doi = {10.5281/zenodo.4414861},
    howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
  }
  ```

### ConvNeXt V1

- **论文**：Liu et al., *A ConvNet for the 2020s*, CVPR 2022
- **仓库**：https://github.com/facebookresearch/ConvNeXt
- **许可证**：MIT
- **引用**：
  ```
  @Article{liu2022convnet,
    author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
    title   = {A ConvNet for the 2020s},
    journal = {CVPR},
    year    = {2022},
  }
  ```

### ConvNeXt V2

- **论文**：Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, CVPR 2023
- **仓库**：https://github.com/facebookresearch/ConvNeXt-V2
- **许可证**：MIT（代码）；CC-BY-NC（预训练权重，本项目未使用）
- **引用**：
  ```
  @article{Woo2023ConvNeXtV2,
    title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
    author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
    year={2023},
    journal={arXiv preprint arXiv:2301.00808},
  }
  ```

> **说明**：本项目从零训练，未加载任何 ImageNet 预训练权重，因此不涉及 CC-BY-NC 的非商业限制。

## 致谢

- 原始论文：Weixin Yang, Lianwen Jin, et al. *DeepWriterID: An End-to-end Online Text-independent Writer Identification System*.
- 数据集：中国科学院自动化研究所 CASIA-OLHWDB 手写数据库。
- ConvNeXt 架构参考：Liu et al., *A ConvNet for the 2020s*, CVPR 2022.
- ConvNeXt V2 架构参考：Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, CVPR 2023.
- timm 库：Ross Wightman, *PyTorch Image Models*.

## 💬 反馈与建议

如果你在使用本项目的过程中遇到任何问题，或者有更好的改进思路，非常欢迎你在 GitHub 上提交 **Issue** 或直接发起 **Pull Request**。

**其他联系方式**：也可以通过作者邮箱 `zhouhao_oss@163.com` 与我沟通。

## 许可证

本项目采用 MIT 许可证。