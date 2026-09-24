"""
ConvNeXt V2 骨干网络：基于 timm 官方实现，适配 64 通道 54×54 输入。
论文：Woo et al., ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders, CVPR 2023.
代码：https://github.com/facebookresearch/ConvNeXt-V2 (MIT)
      https://github.com/huggingface/pytorch-image-models (Apache 2.0)
"""

import torch
import torch.nn as nn
import timm


class ConvNeXtBackbone(nn.Module):
    def __init__(self, num_classes=1019, in_chans=64, feature_dim=512, drop_path_rate=0.1):
        super().__init__()

        # timm 的 convnextv2_tiny：C=(96,192,384,768), B=(3,3,9,3)，与论文一致
        self.backbone = timm.create_model(
            'convnextv2_tiny',
            pretrained=False,       # 从零训练，不加载 ImageNet 权重
            in_chans=in_chans,      # 64 通道
            num_classes=0,          # 去掉分类头，输出 768 维特征
            drop_path_rate=drop_path_rate,
        )

        # 768 维投影到 512 维，对齐 v1.0 的特征维度
        self.feature_head = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x, return_features=False):
        # x: [B, 64, 54, 54]
        feat = self.backbone(x)                # [B, 768]
        features = self.feature_head(feat)     # [B, 512]

        if return_features:
            return features
        return self.classifier(features)