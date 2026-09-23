"""
ConvNeXt Backbone: 替换原始 DCNN，适配 64 通道 54×54 特征图输入。
输出 512 维特征向量，与原始 DCNN 的 fc2 输出对齐。

输入:  [B, 64, 54, 54]
输出:  [B, 512] 或 [B, num_classes]
"""

import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block: DWConv7x7 → GroupNorm → 1x1 → GELU → 1x1 → LayerScale"""

    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return residual + x


class ConvNeXtBackbone(nn.Module):
    """
    适配 64 通道、54×54 输入的轻量级 ConvNeXt。
    输出 512 维特征向量（与原始 DCNN 的 fc2 输出对齐）。

    输入: [B, 64, 54, 54]
    输出: [B, 512] 或 [B, num_classes]
    """

    def __init__(self, num_classes=1019, in_chans=64, feature_dim=512):
        super().__init__()

        # Stem: 4×4 卷积，步长 4  →  54 → 13
        # 注意：in_chans=64，直接接收 dataset 输出的 64 通道特征图
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=4, stride=4),
            nn.GroupNorm(1, 64),
            nn.GELU(),
        )

        # Stage 1: 64 通道，13×13
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
        )

        # Downsample 1: 13 → 6
        self.down1 = nn.Sequential(
            nn.GroupNorm(1, 64),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
        )

        # Stage 2: 128 通道，6×6
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
        )

        # Downsample 2: 6 → 3
        self.down2 = nn.Sequential(
            nn.GroupNorm(1, 128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
        )

        # Stage 3: 256 通道，3×3
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
        )

        # 特征提取头：全局平均池化 + 投影到 512 维
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 64, 54, 54]
            return_features: 如果为 True，返回 512 维特征而不是分类 logits

        Returns:
            logits: [B, num_classes] 或 features: [B, 512]
        """
        x = self.stem(x)          # [B, 64, 13, 13]
        x = self.stage1(x)        # [B, 64, 13, 13]
        x = self.down1(x)         # [B, 128, 6, 6]
        x = self.stage2(x)        # [B, 128, 6, 6]
        x = self.down2(x)         # [B, 256, 3, 3]
        x = self.stage3(x)        # [B, 256, 3, 3]

        features = self.feature_head(x)  # [B, 512]

        if return_features:
            return features

        logits = self.classifier(features)
        return logits