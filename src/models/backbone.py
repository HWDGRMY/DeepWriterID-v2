"""
ConvNeXt V2 骨干网络：基于 timm 官方实现，适配 64 通道 96×96 输入。

论文：Woo et al., ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders, CVPR 2023.
代码：https://github.com/facebookresearch/ConvNeXt-V2 (MIT)
      https://github.com/huggingface/pytorch-image-models (Apache 2.0)

输入： [B, 64, 96, 96]（由 dataset 的 _pad_to_96 生成，原始尺寸 54×54）
输出： [B, 512] 特征向量，或 [B, num_classes] 分类 logits

预训练权重说明：
- 使用 timm 提供的 convnextv2_femto 的 ImageNet-1K 权重（基于 FCMAE 预训练 + 有监督微调）
- 由于输入通道从 3 改为 64，timm 会将第一层卷积重新初始化为随机权重
- 其余所有层加载预训练权重
- 预训练权重许可证：CC-BY-NC 4.0（仅限非商业用途）
"""

import torch
import torch.nn as nn
import timm


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt V2-Femto 骨干，适配 64 通道 96×96 笔迹特征图。

    Femto 配置（论文 Table 14）：
    - 通道：C = (48, 96, 192, 384)
    - Block：B = (2, 2, 6, 2)
    - 参数量：约 5.2M
    - ImageNet-1K Top-1：78.5%
    """

    def __init__(
        self,
        num_classes=1019,
        in_chans=64,
        feature_dim=512,
        drop_path_rate=0.1,
        dropout_head=0.3,
        dropout_classifier=0.4,
        pretrained=True,
    ):
        super().__init__()

        # timm 的 convnextv2_femto
        # pretrained=True 时加载 ImageNet-1K 权重
        # 注意：in_chans=64 时，timm 会把第一层卷积重初始化为随机权重
        self.backbone = timm.create_model(
            'convnextv2_femto',
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        # 384 维投影到 512 维，与 v1.0 DCNN 的特征维度对齐
        self.feature_head = nn.Sequential(
            nn.Linear(384, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_head),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_classifier),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 64, 96, 96] 笔迹特征图
            return_features: True 时返回 512 维特征向量（供下游任务使用）

        Returns:
            logits: [B, num_classes] 或 features: [B, 512]
        """
        feat = self.backbone(x)                  # [B, 384]
        features = self.feature_head(feat)       # [B, 512]

        if return_features:
            return features

        logits = self.classifier(features)
        return logits