# DeepWriterID v2.0：ConvNeXt V2 架构升级与数据流修复

## 项目简介

本项目是 [DeepWriterID-Replication](https://github.com/HWDGRMY/DeepWriterID-Replication)（v1.0）的架构升级版本，在 CASIA-OLHWDB 2.0-2.2（1019 位书写者）上验证 ConvNeXt V2 骨干网络替代原始 DCNN 的效果。

v2.0 在原 v1.0 基础上完成了两件事：

1. **架构升级**：用 ConvNeXt V2-Femto 替换 DCNN 骨干
2. **数据流修复**：修正了 v1.0 中存在的拐点检测阈值 bug，使伪字符切分和 DropSegment 数据增强真正生效

> **v1.0 仓库（DCNN 基线、数据准备、硬件约束等）**：https://github.com/HWDGRMY/DeepWriterID-Replication

> **v2.0 当前状态**：已完成训练与评估

---

## 核心成绩

| 协议 | 类别数 | 页面级准确率 |
| :--- | :--- | :--- |
| **1-test（单次预测）** | 1019 | **96.17%** (980/1019) |
| **20-test（20 次 DropSegment 增强平均）** | 1019 | **98.04%** (999/1019) |

**对比参考**：

| 项目 | 协议 | 类别数 | Page Acc |
| :--- | :--- | :--- | :--- |
| 原论文（NLPR） | 20-test | 187 | 95.72% |
| v1.0（DCNN） | 1-test | 1019 | 95.09% |
| **v2.0（ConvNeXt V2）** | **20-test** | **1019** | **98.04%** |

> **注**：v1.0 的 95.09% 是在「拐点检测阈值 bug 未修复」的数据流下取得的，其伪字符实际退化为「扁行」而非「方形」。v2.0 修复 bug 后，伪字符恢复为方形，与原始论文的方法学对齐。因此 v1.0 与 v2.0 的对比存在数据流差异，读者需注意。

> **关于准确率提升的归因说明**
>
> 本项目的类别数约为原论文的 5.4 倍（1019 vs 187），但准确率的提升（98.04% vs 95.72%）不能直接线性归因于任务难度增加。实际的贡献因素至少包括以下几项：
>
> 1. **模型架构改进**：用 ConvNeXt V2-Femto 替换原始 DCNN，引入大卷积核（7×7 深度卷积）、LayerNorm、GELU、倒瓶颈结构和 GRN 层，提升了特征表达能力。
> 2. **模型参数量增加**：从 DCNN 的 2.6M 增加到 ConvNeXt V2 的 5.2M（约 2 倍），模型容量更大，拟合能力更强。
> 3. **训练数据同步增加**：每位作者的训练页数从原论文的 2 页增加到 4 页，训练样本总量约为原论文的 2 倍，在一定程度上缓解了类别数增加带来的难度。
> 4. **测试协议差异**：原论文使用固定内容测试页 + 20-test 增强；本项目使用随机内容测试页 + 20-test 增强。两者在测试内容与泛化要求上不完全一致。
>
> 因此，上述结果的直接对比存在方法学差异，提升幅度不能简单归因于单一因素。读者需谨慎解读。

---

## 📊 数据集对比说明

本项目使用的数据集与原论文（DeepWriterID）存在重大差异。

| 对比维度 | 原论文（NLPR） | 本项目（CASIA-OLHWDB 2.0-2.2） |
| :--- | :--- | :--- |
| **作者数量** | 187 位书写者 | **1019 位**书写者 |
| **任务类别数** | 187 类 | **1019 类**（5.4 倍） |
| **测试集内容** | 固定内容（同一篇文章） | **内容各不相同**（不同新闻、古诗） |
| **训练样本量** | 较小 | 约 **120 万个**动态增强伪字符 |

---

## 🧠 v2.0 架构升级说明

### 为什么升级？

原始 DCNN 架构（2019 年 DeepWriterID 原论文）使用 3×3 卷积堆叠、BatchNorm、ReLU 和传统瓶颈残差结构。v2.0 用 **ConvNeXt V2-Femto** 替换，引入现代化骨干网络。

### 升级内容

| 模块 | v1.0 | v2.0 |
| :--- | :--- | :--- |
| **骨干网络** | DCNN（2.6M） | **ConvNeXt V2-Femto（5.2M）** |
| **特征维度** | 512 维 | 512 维（对齐） |
| **分类头** | Dropout 0.2 + Linear | Dropout 0.4 + Linear |
| **数据流水线** | 路径签名 + DropSegment | 完全复用（修复后） |
| **训练策略** | Adam + 余弦退火 | **AdamW + Warmup + 余弦退火** |

### ConvNeXt V2 相对传统 CNN 的 6 项升级

| 设计项 | 传统 DCNN | ConvNeXt V2 | 借鉴来源 |
| :--- | :--- | :--- | :--- |
| **卷积核尺寸** | 3×3 堆叠 | 7×7 深度卷积 | Transformer 全局感受野 |
| **归一化层** | BatchNorm | LayerNorm | Transformer 标准 |
| **激活函数** | ReLU | GELU | BERT / GPT |
| **瓶颈结构** | 先降维再升维 | 先升维再降维（4×） | Transformer MLP |
| **下采样方式** | 残差块内 stride=2 | 独立下采样层 | Swin Transformer |
| **通道竞争** | 无 | **GRN 层**（无参数） | V2 新增 |

### 为什么选择 ConvNeXt V2？

1. **性能对标有据可依**：ConvNeXt 原论文已验证其与 Swin Transformer 在 ImageNet、COCO、ADE20K 上性能相当
2. **GRN 层缓解特征坍塌**：论文 Table 14 显示仅加 GRN 即可提升 ImageNet 准确率 0.5~0.9%
3. **部署友好**：后续移植国产 NPU 平台，卷积算子比自注意力算子成熟
4. **消融价值**：可作为「现代化 CNN vs 层次化 Transformer」的对比基准

### Femto 变体的选择

| 变体 | 参数量 | 与 v1.0 DCNN 比值 | 选择 |
| :--- | :--- | :--- | :--- |
| Atto | 3.7M | 1.4× | — |
| **Femto** | **5.2M** | **2.0×** | ✅ 参数量最平衡 |
| Pico | 9.1M | 3.5× | 过拟合风险 |
| Tiny | 28.6M | 11.0× | 远超数据规模 |

---

## ⚡ 训练速度优化历程

本项目在 v2.0 开发过程中，对数据流水线和训练配置做了系统性的性能优化，将单 epoch 时间从最初的 **31 分钟** 压缩到 **约 17 分钟**，云端训练总成本从 **约 73 元** 降至 **约 35 元**。

### 优化前的基线

| 项 | 值 |
| :--- | :--- |
| 单 epoch 时间 | 31 分钟（train 16 + test 15） |
| GPU 利用率 | 脉冲式 20% → 100% → 20% |
| 内存峰值 | 56 GB / 60 GB（濒临 OOM） |
| 崩溃频率 | 每 2-3 轮 OOM Killed |

### 优化过程（按实施顺序）

#### 优化 1：在 `__init__` 缓存切分结果

**问题**：`__getitem__` 每次取一个伪字符，都对整页 4000+ 个点重新做 `detect_corners` + `pseudo_segment`。同一页的 294 个伪字符，等于切分被重复计算 294 次。

**修复**：在 `__init__` 里只切分一次，把每个伪字符的坐标缓存到 `self.char_points_list`。`__getitem__` 直接 O(1) 读取。

**效果**：

| 指标 | 优化前 | 优化后 |
| :--- | :--- | :--- |
| 单次 `__getitem__` | 60 ms | 10-15 ms |
| 训练速度 | 1.77 it/s | **4.9 it/s** |
| 单 epoch | 31 分钟 | 15 分钟 |

**内存代价**：85 万伪字符的坐标缓存约 700 MB，可接受。

#### 优化 2：切分后释放页面级轨迹

**问题**：`self.pages`（4073 页的原始点序列）在切分后不再被使用，但一直占着 5-10 GB 内存。

**修复**：切分完成后立刻 `self.pages = None` + `gc.collect()`。

**效果**：

| 指标 | 优化前 | 优化后 |
| :--- | :--- | :--- |
| 内存峰值 | 56 GB | **41 GB** |
| OOM 风险 | 高 | 消除 |

#### 优化 3：`multiprocessing_context` 从 `spawn` 改为 `fork`

**背景**：PyTorch `DataLoader` 默认在 Windows 用 `spawn`、Linux 用 `fork`。v2.0 早期显式指定了 `spawn`，导致每个 worker 都要重新导入 numpy/cv2/torch/signatory。

**问题**：
- 每个 worker 启动需 3-5 秒，16 个 worker = 1 分钟启动开销
- `spawn` 每轮创建新 semaphore，累积到第 2-3 轮触发 `BrokenPipeError`
- 内存占用比 `fork` 高 30-50%

**修复**：显式指定 `multiprocessing_context='fork'`。

**效果**：

| 指标 | spawn | fork |
| :--- | :--- | :--- |
| 训练速度 | 4.5 it/s | **6.5 it/s** |
| worker 启动 | 每个 3-5 秒 | 几乎瞬时 |
| semaphore 泄漏 | 严重 | 轻微 |
| 内存占用 | 41 GB | **约 40 GB** |

#### 优化 4：`num_workers` 调优

在 fork 模式下，实测不同 worker 数量的吞吐量：

| num_workers | it/s | 结论 |
| :--- | :--- | :--- |
| 14 | 6.2 | 偏少 |
| 15 | 6.4 | 接近最优 |
| **16** | **6.5** | ✅ 甜点 |
| 17 | 6.3 | 上下文切换开销 |
| 18 | 6.2 | 更差 |

**结论**：`num_workers=16` 是 AutoDL 18 vCPU 实例的最优值。再往上加会因为上下文切换而变慢。

#### 优化 5：`batch_size` 调优

| batch_size | it/s | samples/s | 结论 |
| :--- | :--- | :--- | :--- |
| 128 | 7.5 | 960 | CPU 空闲 |
| **192** | **6.5** | **1248** | ✅ 甜点 |
| 256 | 3.6 | 921 | **GPU 成为瓶颈** |

**结论**：`batch_size=192` 是 CPU 与 GPU 负载最平衡的值。用 256 时 GPU 利用率从 55% 飙升到接近满载，但 CPU 供不上数据，整体吞吐反而下降。

### 优化总效果

| 项 | 优化前 | 优化后 | 提升 |
| :--- | :--- | :--- | :--- |
| 单 epoch | 31 分钟 | **17 分钟** | **1.8×** |
| 50 epoch | 26 小时 | **14 小时** | — |
| 云端成本 | 约 73 元 | **约 35 元** | **-52%** |
| 内存峰值 | 56 GB | **41 GB** | -27% |
| 崩溃频率 | 每 2-3 轮 | 无 | — |

---

## 🔬 性能开销分析：修复后对 CPU 单核能力的要求提升

数据流修复带来了性能开销的显著变化，值得单独分析。

### 修复前后的计算量对比

| 项 | 修复前 | 修复后 | 变化 |
| :--- | :--- | :--- | :--- |
| **每页伪字符数** | 约 209（扁行） | 约 294（方形） | **+40%** |
| **训练样本总数** | 85 万 | **120 万** | **+41%** |
| **拐点检测次数** | 约 209 / 页 | 约 294 / 页 | **+40%** |
| **DropSegment 是否执行** | 否（失效） | **是** | 额外 +1 次 detect_corners + pseudo_segment / 样本 |
| **路径签名次数** | 85 万 | 120 万 | **+41%** |
| **单次 `__getitem__` 时间** | 10-15 ms | **约 20-25 ms** | **+60%** |

### 为什么对 CPU 单核能力要求更高

路径签名和 DropSegment 的核心操作是**串行计算**，难以并行化：

1. **`detect_corners`**：逐点计算弯曲值，依赖前 `k` 个和后 `k` 个点，本质上是**串行扫描**。
2. **`pseudo_segment`**：按顺序累加宽度，遇到超过阈值就切分，也是**串行**。
3. **`signatory.signature`**：5 阶迭代积分，逐点递推计算，**无法并行**。
4. **`cv2.polylines` + `equalizeHist`**：单线程绘图与直方图均衡化。

**这些操作的耗时完全由 CPU 的单核性能决定。** 16 个 worker 只是把不同样本分配到不同核心，但每个样本内部的串行链路无法加速。

### 关键结论

> **v2.0 修复后，GPU 利用率仍在 50% 左右（未饱和），CPU 利用率长期在 1600%+（16 核全跑）。瓶颈完全在 CPU 侧的串行数据流水线上。**
>
> **提升吞吐量的唯一有效路径是提高 CPU 单核性能**，而不是堆核心数、堆显存或换更好的 GPU。

---

## 🔧 v2.0 关键修复：数据流修正

### 修复前的问题

v1.0 的代码中存在三个隐蔽 bug，导致数据流与原始论文严重偏离：

**Bug 1：拐点检测阈值不匹配**

- `load_wptt_page` 返回的坐标已被 `×0.1` 缩放（量级 0-720）
- `detect_corners` 的默认阈值 `threshold=180` 是按未缩放量级设计的
- 结果：**检测不到任何拐点** → `pseudo_segment` 的拐点切分机制失效，退化为「按宽度硬切」。每页仍会切出约 209 个伪字符，但每个都是**扁行**（宽度远大于高度），而非原始论文预期的**方形**

**Bug 2：多笔画被虚假连接**

- `render_trajectory_to_bitmap` 用 `cv2.polylines` 一次画完所有点
- `-1`（抬笔分隔符）被过滤后，相邻笔画之间被一条**凭空直线**连接
- 影响：「二」的两横之间会出现一条竖线

**Bug 3：路径签名被 `-1` 污染**

- `signatory.signature` 直接接收包含 `-1` 的轨迹
- `(-1, 0)` 跳变点破坏积分的几何意义

### 修复内容

| 位置 | 修复 |
| :--- | :--- |
| `detect_corners` 调用 | `threshold=180` → `threshold=3` |
| `render_trajectory_to_bitmap` | 按 `-1` 分段绘制，每段独立 `polylines` |
| 签名计算前 | 过滤掉 `-1` 点，只对有效轨迹积分 |
| `WPTTDataset` | 新增 `apply_drop` 参数（训练时 F、20-test 时 T） |
| `WPTTDataset` | 新增 `segments_list` 缓存，避免重复 `detect_corners` |

### 修复效果对比

| 项 | 修复前 | 修复后 |
| :--- | :--- | :--- |
| 每页伪字符数 | 约 209 | 约 294 |
| 伪字符**形状** | **扁行**（宽 ≫ 高） | **方形**（宽 ≈ 高） |
| 拐点检测 | 失败（0 个拐点） | 正常（每页约 2000 个） |
| 切分依据 | 退化为「按宽度硬切」 | 「宽度超过平均高度」 |
| DropSegment | 完全失效 | 生效 |
| 20-test 每轮结果 | 完全相同 | 每轮不同 |

---

## 📉 过拟合判断与方法学局限

### 1. 是否存在过拟合？

**结论：训练后期存在轻微过拟合趋势。96.17% 对应的模型是基于测试集评估选出的最佳权重，由于训练时未单独划分验证集，该准确率存在选择偏差，不能视为无偏泛化估计。但在当前训练条件下，它是测试集上表现最好的模型，且没有出现严重过拟合。**

判断依据如下：

- **页面级准确率在 Epoch 38 达到峰值 96.17%**，这是整个训练过程中测试集表现最好的时刻。
- **Epoch 39–48 的页面级准确率在 94.0%–96.1% 之间小幅波动**，既没有继续上升，也没有出现断崖式下跌。
- **同期训练集字符级准确率在 Epoch 40 达到峰值 17.28% 后开始下滑**（至 Epoch 48 的 16.05%），说明模型已开始记忆训练集中的部分细节，训练集和测试集的差距在扩大。
- **Loss 在 Epoch 41 达到最低 4.6660 后开始回升**（至 Epoch 48 的 4.7535），进一步印证了训练后期出现过拟合趋势。
- **得益于三层正则化（DropPath 0.1 + Dropout 0.3/0.4）、DropSegment 数据增强，以及页面级投票的平滑作用**，测试集准确率没有出现明显退化。
- **训练脚本采用「最佳模型保存」机制**，最终保留的是 Epoch 38 对应的 `convnext_best.pth`，而非最后一轮的权重。因此，轻微过拟合不会影响最终交付模型的质量。

### 2. 方法学局限：未划分独立验证集

由于训练时未划分独立验证集，本项目采用了 **每轮在测试集上评估并保存最佳模型** 的机制。这导致测试集被同时用于：

1. 每轮性能评估；
2. 最佳模型选择（保存 Epoch 38 对应的 `convnext_best.pth`）。

因此，最终报告的 **96.17% 是测试集上的最优选择结果，存在一定乐观偏差，并非严格意义上的无偏泛化估计**。

**为什么没有划分验证集？**

沿用 v1.0 与原论文的做法。每位作者仅有 5 页数据，若划分独立验证集，只能采用 **3 页训练 / 1 页验证 / 1 页测试** 的方式：

- 训练数据将减少 25%，可能导致模型泛化性能下降；
- 验证集规模较小（约 1019 页），反复用于模型选择也容易过拟合验证集。

### 3. 缓解措施：三重评估协议

为部分弥补方法学局限，本项目同时报告了三种评估结果：

| 协议 | 值 | 含义 |
| :--- | :--- | :--- |
| **1-test（best，Epoch 38）** | **96.17%** | 测试集上最优选择结果，存在选择偏差 |
| **1-test（Epoch 50 最终轮）** | **96.07%** | 训练收敛的最终结果，无选择偏差 |
| **20-test（20 次 DropSegment 增强平均）** | **98.04%** | 原论文协议，多重增强后的稳定评估 |

**其中 Epoch 50 的 96.07% 可视为更接近无偏的泛化估计**，而 96.17% 与 98.04% 分别对应「测试集最优选择」与「多重增强平均」两种方法学设定。

### 5. 有记录的轮次指标汇总

| Epoch | Train Loss | Char Acc | Page Acc |
| :--- | :--- | :--- | :--- |
| 1 | 5.2916 | 11.32% | 67.22% |
| 5 | 5.6004 | 6.56% | 67.71% |
| 10 | 5.3991 | 8.30% | 73.99% |
| 15 | 5.2299 | 10.01% | 78.61% |
| 20 | 5.5016 | 7.37% | 80.27% |
| 25 | 5.2636 | 9.81% | 86.56% |
| 28 | 5.1155 | 11.46% | 92.84% |
| 30 | 5.0139 | 12.69% | 93.92% |
| 33 | 4.8647 | 14.56% | **95.88%** |
| 35 | 4.7776 | 15.71% | 94.80% |
| **38** | **4.6932** | **16.90%** | **96.17%** ← 峰值 |
| 40 | 4.6699 | **17.20%** | 96.07% |
| 42 | 4.6684 | 17.24% | 95.88% |
| 44 | 4.6794 | 17.08% | 95.78% |
| 46 | 4.7079 | 16.74% | 95.88% |
| 48 | 4.7535 | 16.05% | 94.01% |
| 50 | — | — | 96.07% |

> **观察**：Page acc 在 Epoch 38 达到峰值后开始波动下降；Char acc 在 Epoch 40 达到峰值后开始下滑；Loss 在 Epoch 41 之后回升。三者共同指向训练后期轻微过拟合。

### 6. 展望

未来如有条件，可从以下方向改进评估的严谨性：

- **扩充数据规模**，使每位作者拥有更多页面，从而支持训练集 / 验证集 / 测试集的严格划分；
- **引入独立验证集**用于模型选择，使测试集完全隔离于训练与调参过程，从而获得无偏的泛化估计；
- **跨数据集验证**：在 NLPR 或 HIT-MW 等其他中文手写数据集上评估模型的跨域泛化能力。

---

## 🔍 训练策略说明：基于旧数据流权重的继续训练

### 为什么需要交代这一点

本项目 v2.0 的最终成绩（1-test 96.17% / 20-test 98.04%）**并非从零训练得到**，而是**加载了「旧数据流（扁行伪字符）」下训练的 ConvNeXt V2 权重，在「修复后数据流（方形伪字符）」上继续训练**的结果。

这一点必须明确说明，否则读者无法正确解读成绩的归因。

### 完整训练历程

| 阶段 | 数据流 | 伪字符形状 | 训练起点 | 达到的最佳成绩 |
| :--- | :--- | :--- | :--- | :--- |
| **阶段 1** | 旧（threshold=180） | **扁行** | 随机初始化（从零） | **1-test 96.47%**（Epoch 46） |
| **阶段 2** | 新（threshold=3） | **方形** | **加载阶段 1 权重** | **1-test 96.17%**（Epoch 38） |

**注意**：阶段 2 的 96.17% **略低于**阶段 1 的 96.47%，而非「修复后更高」。这说明两件事：

1. **方形伪字符确实更难**——单样本信息量更小（一个字 vs 一行），分类难度上升
2. **旧权重并非无效**——它已经学到了通用的笔迹风格特征，能快速适配新数据流

### 对比与误差分析

| 项 | 阶段 1（旧数据流）  | 阶段 2（新数据流） |
| :--- |:--------------------| :--- |
| **伪字符形状** | 扁行（宽 ≫ 高）     | 方形（宽 ≈ 高） |
| **每页伪字符数** | 约 209              | 约 294 |
| **训练样本数** | 85 万               | 120 万 |
| **DropSegment** | 完全失效            | 生效 |
| **Epoch 1 起点** | 从零（loss ≈ 6.93） | 加载旧权重（loss ≈ 5.29） |
| **最佳 1-test** | **96.47%**          | **96.17%** |
| **20-test** |  96.47%（每轮相同） | **98.04%（每轮不同）** |

**误差范围分析**：

- 阶段 1 与阶段 2 的 1-test 差距为 **0.30%**
- 该差距在随机波动范围内（两次独立训练的波动标准差约 0.3-0.5%）
- **不能断言「修复数据流后性能下降」**——两者在统计上等价

### 20-test 的有效性验证

**修复前**（旧数据流 + DropSegment 失效）：

| 轮次 | 逐轮 Page Acc | 累积 Page Acc |
| :--- | :--- | :--- |
| 1 | 96.47% | 96.47% |
| 2 | 96.47% | 96.47% |
| ... | ... | ... |
| 20 | 96.47% | 96.47% |

**20 轮结果完全相同 → 证明 DropSegment 未生效 → 20-test 退化为「重复 1-test」**

**修复后**（新数据流 + DropSegment 生效）：

| 轮次 | 逐轮 Page Acc | 累积 Page Acc |
| :--- | :--- | :--- |
| 1 | 97.64% | 97.64% |
| 2 | 97.15% | 97.74% |
| 5 | 96.96% | 97.94% |
| 10 | 97.84% | **98.23%** |
| 15 | 97.25% | 98.14% |
| 20 | 97.64% | **98.04%** |

**20 轮结果每轮不同 → 证明 DropSegment 真正生效 → 20-test 是有效的多轮增强平均**

**累积列的非单调性**（如第 10 轮 98.23% → 第 11 轮 98.04%）也符合概率平均的预期——它不是「准确率平均」，而是「多轮概率平均后再投票」的集成结果。

---

## ⚙️ 训练配置

### 硬件环境

| 项目 | 配置 |
| :--- | :--- |
| GPU | NVIDIA RTX 4090 D（24GB） |
| CPU | AMD EPYC 9754（18 vCPU） |
| RAM | 60 GB |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| timm | 1.0.30 |

### 超参数

| 超参数 | 值 | 依据 |
| :--- | :--- | :--- |
| **batch_size** | 192 | 平衡 CPU 与 GPU |
| **lr_base** | 0.000667 | 实际 lr = 5e-4 |
| **optimizer** | AdamW(β₁=0.9, β₂=0.999) | ConvNeXt V2 论文 |
| **weight_decay** | 0.05 | 论文推荐 |
| **warmup_epochs** | 1 | 从零训练 |
| **epochs** | 50 | 余弦退火 |
| **drop_path** | 0.1 | ConvNeXt V2 Atto/Nano |
| **dropout_head** | 0.3 | 原 DeepWriterID 浅层 |
| **dropout_classifier** | 0.4 | 原 DeepWriterID 中层 |
| **num_workers** | 16 | fork 模式 |
| **multiprocessing_context** | fork | Linux 下更快 |

### 20-test 增强评估

| 轮次 | 逐轮 Page Acc | 累积 Page Acc |
| :--- | :--- | :--- |
| 1 | 97.64% | 97.64% |
| 5 | 96.96% | 97.94% |
| 10 | 97.84% | **98.23%** |
| 15 | 97.25% | 98.14% |
| 20 | 97.64% | **98.04%** |

**20-test 最终 = 98.04%（累积 20 轮概率平均后的页面准确率）**

---

## ⚠️ 平台适配提醒：`fork` 仅支持 Linux / macOS

本项目在云端（Ubuntu 22.04）使用了 `multiprocessing_context='fork'`，这是 Linux 和 macOS 的默认机制，**性能显著优于 `spawn`**。

**但 `fork` 在 Windows 上完全不可用**（Windows 没有 fork 系统调用）。如果你在 Windows 上运行本项目，必须：

1. 把 `trainer.py` 和 `evaluator.py` 里的 `multiprocessing_context='fork'` 改回 `'spawn'`
2. 或者直接删除该参数，让 PyTorch 自动选择平台默认值

```python
# 跨平台写法
import sys
if sys.platform == 'win32':
    ctx = 'spawn'
else:
    ctx = 'fork'

train_loader = DataLoader(
    train_dataset,
    num_workers=16,
    multiprocessing_context=ctx,
    # ...
)
```

**预期差异**：

| 平台 | 上下文 | 单 epoch |
| :--- | :--- | :--- |
| Linux（云端） | fork | **17 分钟** |
| Windows（本地） | spawn | 约 25 分钟 |

本地 Windows 因 `spawn` 开销更大，实际训练速度会慢 30-40%，这是平台固有限制，非代码问题。

### 补充说明：为什么 `fork` 对训练结果无影响

1. **PyTorch DataLoader 在 worker 启动时会用 `base_seed + worker_id` 重新设置随机种子**，DropSegment 的随机性完全保留
2. **CUDA context 不会被 `fork` 破坏**，因为 worker 只负责 CPU 数据处理，不碰 GPU
3. 实测：`fork` 和 `spawn` 下训练相同轮次，最终准确率差异 < 0.1%

**`fork` 是 Linux 平台下的性能最优解，且不牺牲结果可复现性。**

---

## 📁 项目目录结构

```text
DeepWriterID-v2/
├── configs/
│   └── convnext.yaml               # ConvNeXt V2 超参数
├── data/
│   ├── raw/                        # 原始 .wptt 轨迹
│   └── features/metadata.csv       # 数据划分
├── src/
│   ├── data/
│   │   ├── loader.py               # 解析 .wptt
│   │   └── dataset.py              # WPTTDataset（含 apply_drop 开关）
│   ├── models/
│   │   ├── backbone.py             # ConvNeXt V2-Femto
│   │   └── dcnn.py                 # DCNN（保留作为基线）
│   ├── preprocessing/
│   │   ├── corner.py               # 拐点检测（threshold=3）
│   │   └── segmentation.py         # 伪字符切分
│   ├── features/
│   │   └── path_signature.py       # 路径签名（含分段绘制修复）
│   ├── augmentation/
│   │   └── drop_segment.py         # DropSegment（片段缓存版）
│   ├── training/
│   │   └── trainer.py              # 训练引擎
│   └── evaluation/
│       └── evaluator.py            # 页面级投票评估（1-test + 20-test）
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── evaluate_v2.py              # 测试 v2 模型
├── outputs/
│   ├── logs/
│   └── checkpoints/
├── environment.yml
├── setup.py
├── README.md
└── LICENSE
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
conda env create -f environment.yml
conda activate deepwriterid
pip install timm
```

或使用 pip：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm pyyaml tqdm pandas opencv-python matplotlib numpy scikit-learn
pip install signatory --no-build-isolation
```

### 2. 数据准备

从 CASIA 官网下载 OLHWDB 2.0-2.2：

```bash
python scripts/preprocess.py
```

### 3. 训练

```bash
python scripts/train.py --config configs/convnext.yaml
```

**特性**：
- Warmup + 余弦退火（iteration 级调度）
- 断点续训（模型、优化器、调度器完整保存）
- 早停（连续 10 轮未提升）
- 页面级投票评估（每轮 1-test）

### 4. 评估

**1-test（与训练每轮测试一致）**：

```bash
python scripts/evaluate_v2.py --ckpt outputs/checkpoints/convnext_best.pth --n_test 1
```

**20-test（原论文协议）**：

```bash
python scripts/evaluate_v2.py --ckpt outputs/checkpoints/convnext_best.pth --n_test 20
```

---

## 📦 模型文件说明

模型权重不包含在 GitHub 仓库中，原因：

1. **算力成本**：训练消耗约 12 小时云端算力
2. **模型资产保护**：在 1019 类 CASIA 数据集上达到 98.04%（20-test）

### 📎 如何获取

- **GitHub Issues**：[新建 Issue](https://github.com/HWDGRMY/DeepWriterID-v2/issues) 说明用途
- **邮件**：`zhouhao_oss@163.com`，附身份、用途及具体场景

> **注意**：模型文件仅限申请用途使用，请勿二次分发。

---

## 📜 第三方代码与引用

### timm (PyTorch Image Models)

- **仓库**：https://github.com/huggingface/pytorch-image-models
- **许可证**：Apache License 2.0
- **引用**：
  ```
  @misc{rw2019timm,
    author = {Ross Wightman},
    title = {PyTorch Image Models},
    year = {2019},
    doi = {10.5281/zenodo.4414861}
  }
  ```

### ConvNeXt V1

- **论文**：Liu et al., *A ConvNet for the 2020s*, CVPR 2022
- **仓库**：https://github.com/facebookresearch/ConvNeXt
- **许可证**：MIT

### ConvNeXt V2

- **论文**：Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, CVPR 2023
- **仓库**：https://github.com/facebookresearch/ConvNeXt-V2
- **许可证**：MIT（代码）；CC-BY-NC（预训练权重，本项目未使用）

### 原 DeepWriterID

- **论文**：Yang et al., *DeepWriterID: An End-to-end Online Text-independent Writer Identification System*, 2015

### Dropout

- **论文**：Hinton et al., *Improving neural networks by preventing co-adaptation of feature detectors*, 2012

---

## 🙏 致谢

- 原始论文：Weixin Yang, Lianwen Jin, et al. *DeepWriterID: An End-to-end Online Text-independent Writer Identification System*.
- 数据集：中国科学院自动化研究所 CASIA-OLHWDB 手写数据库。
- ConvNeXt 架构参考：Liu et al., CVPR 2022.
- ConvNeXt V2 架构参考：Woo et al., CVPR 2023.
- timm 库：Ross Wightman.
- Dropout 理论基础：Hinton et al., 2012.

---

## 💬 反馈与建议

如果你在使用本项目的过程中遇到任何问题，或者有更好的改进思路，非常欢迎你在 GitHub 上提交 **Issue** 或直接发起 **Pull Request**。

**其他联系方式**：也可以通过作者邮箱 `zhouhao_oss@163.com` 与我沟通。

---

## 📄 许可证

本项目采用 MIT 许可证。