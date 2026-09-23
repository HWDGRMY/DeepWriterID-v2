"""
ConvNeXt 架构升级训练入口。
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import WPTTDataset
from src.models.backbone import ConvNeXtBackbone
from src.training.trainer import Trainer


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/convnext.yaml')
    parser.add_argument('--resume', action='store_true', help='从断点继续训练')
    args = parser.parse_args()

    config = load_config(args.config)

    # 设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # 随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 数据
    print("[INFO] 加载训练集...")
    train_dataset = WPTTDataset(
        metadata_file='data/features/metadata.csv',
        split='Train',
    )
    print("[INFO] 加载测试集...")
    val_dataset = WPTTDataset(
        metadata_file='data/features/metadata.csv',
        split='Test',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )

    # 模型
    model = ConvNeXtBackbone(
        num_classes=config['model']['num_classes'],
        feature_dim=config['model']['feature_dim'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] ConvNeXtBackbone 参数量: {total_params / 1e6:.2f}M")

    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # 断点续训
    if args.resume:
        latest_path = os.path.join(
            config['checkpoint']['save_dir'],
            config['checkpoint']['save_latest'],
        )
        if os.path.exists(latest_path):
            trainer.load_checkpoint(latest_path)
            print(f"[INFO] 已从 {latest_path} 恢复训练")
        else:
            print(f"[WARN] 未找到断点文件 {latest_path}，从头开始训练")

    trainer.fit()


if __name__ == '__main__':
    main()