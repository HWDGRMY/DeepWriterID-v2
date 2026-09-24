# DeepWriterID v2.0：ConvNeXt V2 架构升级

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

---

## 🧠 v2.0 架构升级说明

### 为什么升级？

原始 DCNN 架构（2019 年 DeepWriterID 原论文）使用 3×3 卷积堆叠、BatchNorm、ReLU 和传统瓶颈残差结构。这套范式在 2020 年 Vision Transformer（ViT）出现后，在学术界一度被认为「落后一代」。

2022 年 ConvNeXt 论文（Liu et al., CVPR 2022）提出了一个反直觉的发现：**Vision Transformer 的成功，很大程度上不是因为自注意力机制本身，而是因为它引入的一系列设计决策**（大卷积核、LayerNorm、GELU、倒瓶颈、DropPath）。把这些设计决策「翻译」回纯卷积网络，ConvNet 就能重新与 Transformer 竞争。

ConvNeXt V2（Woo et al., CVPR 2023）在此基础上进一步引入了 **GRN 层**（Global Response Normalization），通过增强通道间竞争来缓解深度网络在细粒度分类中的特征坍塌问题。

### 升级内容

| 模块 | v1.0 | v2.0 |
| :--- | :--- | :--- |
| **骨干网络** | DCNN（5 层卷积 + 3 层全连接） | ConvNeXt V2-Femto |
| **特征维度** | 512 维 | 512 维（对齐 v1.0） |
| **分类头** | Dropout 0.2 + Linear | Dropout 0.4 + Linear |
| **数据流水线** | 路径签名 + DropSegment | 完全复用 v1.0（不变） |
| **训练策略** | Adam + 余弦退火（epoch 级） | AdamW + Warmup + 余弦退火（iteration 级） |

---

## 🔬 ConvNeXt V2 详解

### ConvNeXt V2 相对传统 CNN 的 6 项关键升级

| 设计项 | 传统 DCNN（v1.0） | ConvNeXt V2（v2.0） | 借鉴来源 |
| :--- | :--- | :--- | :--- |
| **卷积核尺寸** | 3×3 堆叠 | 7×7 深度卷积 | Transformer 的全局感受野 |
| **归一化层** | BatchNorm | LayerNorm | Transformer 标准配置 |
| **激活函数** | ReLU | GELU | BERT / GPT 系列 |
| **瓶颈结构** | 先降维再升维 | 先升维再降维（4×） | Transformer MLP 模块 |
| **下采样方式** | 残差块内 stride=2 | 独立下采样层 | Swin Transformer |
| **通道竞争** | 无 | **GRN 层**（无参数） | V2 新增，增强特征多样性 |

### 为什么选择 ConvNeXt V2？

本任务的骨干选型基于以下四点考量：

1. **性能对标有据可依**：ConvNeXt 原论文（Liu et al., CVPR 2022）已系统验证，ConvNeXt 与 Swin Transformer 在 ImageNet 分类上性能相当，在 COCO 检测和 ADE20K 分割上甚至更优。选择 ConvNeXt 不会在性能上牺牲。

2. **GRN 层缓解特征坍塌**：ConvNeXt V2（Woo et al., CVPR 2023）引入的 GRN 层通过增强通道间竞争，专门解决深度网络在细粒度分类任务中的特征坍塌问题。论文 Table 14 显示，仅加 GRN 层即可使监督训练下的 ImageNet 准确率提升 0.5~0.9%。这对本任务的 1019 类分类尤为关键。

3. **部署友好**：后续需将模型移植至国产 NPU（麒麟 + 昇腾）平台。昇腾 CANN 对卷积算子的支持远比自注意力算子成熟——卷积可直接调用硬件加速，自注意力需要额外的动态 shape 处理和内存调度。ConvNeXt 的纯卷积架构能显著降低移植复杂度。

4. **消融价值**：ConvNeXt 与 Swin Transformer 的对比，可作为论文中「现代化 CNN vs 层次化 Transformer」在细粒度笔迹分类任务上的消融实验，为后续工作提供基准。

> **注**：我们将在未来工作中补充 Swin Transformer 作为对照骨干的实验对比，以进一步验证骨干选型的合理性。

### Femto 变体的选择理由

ConvNeXt V2 提供 8 种规模（Atto / Femto / Pico / Nano / Tiny / Base / Large / Huge）。本任务选用 **Femto**：

| 变体 | 参数量 | 与 v1.0 DCNN 的比值 | 选择理由 |
| :--- | :--- | :--- | :--- |
| Atto | 3.7M | 1.4× | 容量偏小，准确率有下降风险 |
| **Femto** | **5.2M** | **2.0×** | ✅ **参数量最平衡** |
| Pico | 9.1M | 3.5× | 85 万样本下开始有过拟合苗头 |
| Tiny | 28.6M | 11.0× | 参数量远超数据规模，过拟合风险极高 |

**Femto 的完整参数量为 5.62M**（骨干 5.2M + 384→512 投影 0.2M + 1019 类分类头 0.52M），与 v1.0 的 DCNN（2.6M）处于同一数量级，但具备全部现代化设计。

### 输入输出规格

```
输入： [B, 64, 96, 96]     ← 由 dataset 的 _pad_to_96 从原始 54×54 补齐
   ↓ ConvNeXt V2-Femto 骨干
中间： [B, 384]            ← 全局平均池化后的特征向量
   ↓ 特征投影层
输出： [B, 512]            ← 与 v1.0 DCNN 的特征维度对齐
   ↓ 分类头
最终： [B, 1019]           ← 1019 位书写者的分类 logits
```

**与 v1.0 保持 512 维特征对齐**是刻意的设计决策——这样后续的伪时序融合、拼接检测、生成检测等下游模块，可以直接复用 v1.0 已建立的接口规范。

---

## 📦 预训练权重说明

本项目加载了 timm 提供的 `convnextv2_femto` 的 ImageNet-1K 预训练权重（模型名 `convnextv2_femto.fcmae_ft_in1k`）。

### 权重来源

该权重由 Facebook Research 团队在 ConvNeXt V2 论文中发布，制作流程为：

1. 先用 **FCMAE**（Fully Convolutional Masked Autoencoder）在 ImageNet-1K 上做自监督预训练
2. 再在 ImageNet-1K 上做有监督微调

因此该权重**同时包含** FCMAE 自监督学习和 ImageNet 有监督学习的知识。

### 通道适配

由于本任务的输入是 **64 通道**笔迹特征图，而原始权重是 **3 通道** RGB 图像，timm 的 `create_model(in_chans=64)` 会：

- 将第一层卷积（stem）从 3 通道改为 64 通道，**该层随机初始化**
- 其余所有层正常加载预训练权重

**第一层卷积约占骨干参数量的 1%**，其余 99% 的参数仍来自预训练。

### 许可证说明

`convnextv2_femto.fcmae_ft_in1k` 权重的许可证为 **CC-BY-NC 4.0**（非商业性使用）。本项目仅用于学术研究，符合许可证要求。如需商业使用，请自行替换为其他权重或从零训练。

---

## 🛡️ 正则化设计：三层防护

ConvNeXt V2 相比 v1.0 的 DCNN，**天然更容易过拟合**，原因有三：

1. 参数量翻倍（2.6M → 5.2M）
2. 失去 BatchNorm 的隐式正则化（改用 LayerNorm）
3. GELU 比 ReLU 更平滑，激活更密集，模型容量更大

为此，本项目设计了**三层防护**，每一层都有明确的论文依据：

| 位置 | 工具 | 值 | 论文依据 |
| :--- | :--- | :--- | :--- |
| **骨干残差块** | DropPath | 0.1 | ConvNeXt V2 论文 Table 9，Atto/Nano 使用 0.1 |
| **feature_head 后** | Dropout | 0.3 | 原 DeepWriterID 论文 5.4 节，浅层全连接 Dropout 0.3 |
| **classifier 前** | Dropout | 0.4 | 原 DeepWriterID 论文 5.4 节，中层全连接 Dropout 0.4 |

### 为什么 DropPath 用 0.1 而不用论文默认的 0.0？

ConvNeXt V2 论文给 Femto 的 DropPath 是 **0.0**，但那是为 ImageNet 128 万张独立图像设计的配置。本任务的有效样本量远小于 ImageNet——85 万伪字符来自约 4000 个原始页面，同一页的伪字符在风格上高度相关。因此，适当加大 DropPath 至 0.1 是必要的调整。

### 为什么 Dropout 用 0.3 和 0.4 而不用原论文的 0.5？

Hinton 的 Dropout 原论文（2012）第 5 页明确指出：

> For fully connected layers, dropout in all hidden layers works better than dropout in only one hidden layer and **more extreme probabilities tend to be worse**, which is why we have used 0.5 throughout this paper.

原 DeepWriterID 论文在 187 类任务上使用 0.3、0.4、0.5、0.5 的逐层递增 Dropout（DCNN 有三个连续全连接层）。ConvNeXt V2 只有一层分类头（投影层 + 分类层），因此我们将原论文的浅层强度 0.3 映射到投影层、中层强度 0.4 映射到分类层。0.5 因对应层不存在而略去。

---

## ⚙️ 训练超参数设置

本项目的训练配置参考 ConvNeXt V2 论文 Table 9（Atto/Femto/Pico/Nano 小模型配置），并按 batch size 做线性缩放：

| 超参数 | 论文值（Femto, batch=1024） | 本项目值（batch=128） | 依据 |
| :--- | :--- | :--- | :--- |
| **优化器** | AdamW(β₁=0.9, β₂=0.999) | 同 | 论文指定 |
| **初始学习率** | 2e-4 | 2.5e-5 | 线性缩放规则 |
| **权重衰减** | 0.05 | 0.05 | 论文推荐 |
| **Warmup** | 0 epoch | 1 epoch | 预训练已提供良好初始化，短 warmup 稳定 |
| **学习率调度** | 余弦退火 | Warmup + 余弦退火 | iteration 级 step |
| **DropPath** | 0.0 | 0.1 | 小数据集需要正则化 |
| **Dropout** | 无 | 0.3 + 0.4 | 对齐原 DeepWriterID 论文 |

> **线性缩放规则**：论文的 `lr = base_lr × batch_size / 256`。本项目 batch=128，因此 `lr = 5e-5 × 128 / 256 = 2.5e-5`。

---

## ⚠️ 与 ConvNeXt V2 论文的关键差异

本项目的核心贡献是**笔迹任务的双路融合与攻防闭环**，而非骨干网络的复现。骨干部分使用 ConvNeXt V2 论文发布的模型与权重，但有以下四点差异：

1. **未复现 FCMAE 预训练过程**：ConvNeXt V2 论文的核心贡献是 FCMAE（全卷积掩码自编码器）+ GRN 的协同设计。本项目**加载了论文发布的预训练权重**（该权重由 FCMAE + ImageNet 微调得到），但**未自行复现** FCMAE 训练框架（800 epoch 的自监督预训练需要大规模算力）。

2. **输入规格适配**：将输入从 3 通道 224×224 改为 64 通道 96×96，以匹配笔迹伪字符的路径签名特征图。

3. **特征维度对齐**：在骨干输出后增加 384→512 投影层，与 v1.0 的特征维度对齐，便于下游模块复用。

4. **超参数调整**：论文 Table 9 的 Femto 配置用于「ImageNet 预训练后微调」，本项目在加载预训练权重的基础上增加了短 warmup 和更大的 DropPath，以适应笔迹任务的样本特性。

**因此，本项目可以表述为「使用了 ConvNeXt V2 的完整模型（含预训练权重）」，但不能表述为「复现了 ConvNeXt V2 的预训练方法」。**

---

## 📁 项目目录结构

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
│   ├── features/                       # 特征提取
│   ├── augmentation/                   # 数据增强
│   ├── training/                       # 训练引擎
│   │   └── trainer.py                  # 核心训练循环
│   ├── evaluation/                     # 评估引擎
│   │   └── evaluator.py                # 页面级投票准确率计算
│   └── utils/                          # 工具函数
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

---

## 🚀 快速开始

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

**训练脚本特性**：
- **预训练权重加载**：从 timm 自动下载 `convnextv2_femto.fcmae_ft_in1k`（首次运行约 20MB）
- **Warmup + 余弦退火**：LinearLR + CosineAnnealingLR，iteration 级 step
- **断点续训**：完整保存/恢复模型、优化器、调度器状态
- **早停**：连续 10 次页面级验证未提升则停止
- **页面级投票评估**：每 5 轮评估一次，与 v1.0 保持一致

### 4. 评估模型

加载最佳模型并进行完整的页面级测试：

```bash
python scripts/evaluate.py --config configs/convnext.yaml
```

---

## 📦 模型文件说明（不公开）

本项目的预训练模型文件（`convnext_best.pth`）**不包含在本 GitHub 仓库中**，原因与 v1.0 相同：算力成本高、模型资产需保护。

### 📎 如何获取模型文件？

- **通过 GitHub Issues 提交申请**  
  请访问 [Issues 页面](https://github.com/HWDGRMY/DeepWriterID-v2/issues) 新建 Issue，简要说明使用目的。
- **通过邮件直接联系作者**  
  发送邮件至 **`zhouhao_oss@163.com`**，请附上身份、用途及具体场景。

> **注意**：模型文件仅限申请用途使用，请勿二次分发。

---

## 📜 第三方代码与引用

本项目在静态分支中使用了以下开源实现，特此致谢并声明引用：

### timm (PyTorch Image Models)

- **仓库**：https://github.com/huggingface/pytorch-image-models
- **许可证**：Apache License 2.0
- **用途**：提供 `convnextv2_femto` 骨干网络及其 ImageNet-1K 预训练权重的 PyTorch 实现
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
- **许可证**：MIT（代码）；CC-BY-NC 4.0（预训练权重）
- **引用**：
  ```
  @article{Woo2023ConvNeXtV2,
    title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
    author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
    year={2023},
    journal={arXiv preprint arXiv:2301.00808},
  }
  ```

### 原 DeepWriterID

- **论文**：Yang et al., *DeepWriterID: An End-to-end Online Text-independent Writer Identification System*
- **用途**：DropSegment 数据增强、路径签名特征、页面级投票评估逻辑
- **引用**：
  ```
  @article{yang2015deepwriterid,
    title={DeepWriterID: An End-to-end Online Text-independent Writer Identification System},
    author={Weixin Yang, Lianwen Jin, Manfei Liu},
    journal={arXiv preprint arXiv:1508.04945},
    year={2015}
  }
  ```

### Dropout

- **论文**：Hinton et al., *Improving neural networks by preventing co-adaptation of feature detectors*, 2012
- **用途**：Dropout 正则化的理论依据，本项目的 Dropout 强度设置参考该论文
- **引用**：
  ```
  @article{hinton2012improving,
    title={Improving neural networks by preventing co-adaptation of feature detectors},
    author={Hinton, Geoffrey E and Srivastava, Nitish and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan R},
    journal={arXiv preprint arXiv:1207.0580},
    year={2012}
  }
  ```

---

## 🙏 致谢

- 原始论文：Weixin Yang, Lianwen Jin, et al. *DeepWriterID: An End-to-end Online Text-independent Writer Identification System*.
- 数据集：中国科学院自动化研究所 CASIA-OLHWDB 手写数据库。
- ConvNeXt 架构参考：Liu et al., *A ConvNet for the 2020s*, CVPR 2022.
- ConvNeXt V2 架构参考：Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, CVPR 2023.
- timm 库：Ross Wightman, *PyTorch Image Models*.
- Dropout 理论基础：Hinton et al., *Improving neural networks by preventing co-adaptation of feature detectors*, 2012.

---

## 💬 反馈与建议

如果你在使用本项目的过程中遇到任何问题，或者有更好的改进思路，非常欢迎你在 GitHub 上提交 **Issue** 或直接发起 **Pull Request**。

**其他联系方式**：也可以通过作者邮箱 `zhouhao_oss@163.com` 与我沟通。

---

## 📄 许可证

本项目采用 MIT 许可证（代码部分）。使用的第三方预训练权重 `convnextv2_femto.fcmae_ft_in1k` 遵循 CC-BY-NC 4.0 许可证，仅限非商业使用。