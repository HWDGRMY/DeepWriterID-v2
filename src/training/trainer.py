"""
ConvNeXt V2 训练器：加载 ImageNet 预训练权重，保留 v1.0 的页面级投票评估逻辑。

关键设计：
- 预训练：支持从本地文件加载权重（避免云端无法访问 HuggingFace）
- 优化器：AdamW(β₁=0.9, β₂=0.999)
- 学习率：线性缩放 lr = lr_base × batch_size / 256
- 调度器：Warmup（1 epoch）+ 余弦退火，iteration 级 step
- 评估：每 EVAL_FREQUENCY 轮做一次页面级投票
- 早停：连续 PATIENCE 次验证未提升则停止
- 断点续训：完整保存/恢复模型、优化器、调度器状态
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import pandas as pd
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import WPTTDataset
from src.models.backbone import ConvNeXtBackbone
from src.evaluation.evaluator import evaluate_page_level


def _pad_to_96(tensor):
    """v1.0 继承：54×54 → 96×96。"""
    pad = (21, 21, 21, 21)
    return torch.nn.functional.pad(tensor, pad, mode='constant', value=0)


def train(config):
    # ========== 读取超参数 ==========
    BATCH_SIZE = config.get('batch_size', 128)
    EPOCHS = config.get('epochs', 50)
    LR_BASE = config.get('lr_base', 5e-5)
    INITIAL_LR = LR_BASE * BATCH_SIZE / 256
    WEIGHT_DECAY = config.get('weight_decay', 0.05)
    WARMUP_EPOCHS = config.get('warmup_epochs', 1)
    EVAL_FREQUENCY = config.get('eval_frequency', 5)
    PATIENCE = config.get('patience', 10)
    DROP_PATH = config.get('drop_path', 0.1)
    DROPOUT_HEAD = config.get('dropout_head', 0.3)
    DROPOUT_CLASSIFIER = config.get('dropout_classifier', 0.4)
    PRETRAINED = config.get('pretrained', True)
    PRETRAINED_PATH = config.get('pretrained_path', None)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"🔧 配置: batch={BATCH_SIZE}, lr={INITIAL_LR:.6f}, "
          f"pretrained={PRETRAINED}, warmup={WARMUP_EPOCHS}, "
          f"drop_path={DROP_PATH}, dropout_head={DROPOUT_HEAD}, "
          f"dropout_classifier={DROPOUT_CLASSIFIER}")

    # ========== 数据路径 ==========
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metadata_file = os.path.join(BASE_DIR, 'data', 'features', 'metadata_small.csv')

    print("正在加载数据集 ...")
    full_df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    full_df['writer_id'] = full_df['writer_id'].astype(str)
    all_writer_ids = sorted(full_df['writer_id'].unique())
    global_label_map = {wid: idx for idx, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)

    # ========== 数据增强与 Dataset ==========
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        _pad_to_96
    ])

    train_dataset = WPTTDataset(metadata_file, split='Train', transform=train_transform)
    print(f"📊 训练集样本数: {len(train_dataset)}")
    print(f"📊 作者（类别）总数: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config.get('num_workers', 8),
        pin_memory=True,
        prefetch_factor=2,
        multiprocessing_context='spawn'
    )

    # ========== 模型（加载预训练权重） ==========
    model = ConvNeXtBackbone(
        num_classes=num_classes,
        in_chans=64,
        feature_dim=512,
        drop_path_rate=DROP_PATH,
        dropout_head=DROPOUT_HEAD,
        dropout_classifier=DROPOUT_CLASSIFIER,
        pretrained=PRETRAINED,
        pretrained_path=PRETRAINED_PATH,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数量: {total_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()

    # ========== 优化器与调度器 ==========
    optimizer = optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    cosine_steps = max((EPOCHS - WARMUP_EPOCHS) * steps_per_epoch, 1)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # ========== 路径与状态 ==========
    log_path = os.path.join(BASE_DIR, 'outputs', 'logs', 'convnext_training_log.csv')
    latest_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'convnext_latest.pth')
    best_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'convnext_best.pth')
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    start_epoch = 1
    best_page_acc = 0.0
    no_improve_count = 0

    # ========== 断点续训 ==========
    if os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_page_acc = checkpoint.get('best_acc', 0.0)
        print(f"✅ 断点恢复: Epoch {start_epoch}, Best Page Acc: {best_page_acc:.4f}")
    else:
        print("🆕 未找到检查点，将从 Epoch 1 开始全新训练。")

    # ========== 训练主循环 ==========
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()                       # iteration 级调度

            running_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()

            lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_train/total_train:.4f}",
                'lr': f"{lr:.6f}",
            })

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        print(f"Epoch {epoch} Train | Loss: {epoch_loss:.4f} | Char Acc: {epoch_acc:.4f}")

        # ========== 页面级评估（每 EVAL_FREQUENCY 轮一次） ==========
        test_page_acc = 0.0
        if epoch % EVAL_FREQUENCY == 0:
            test_page_acc = evaluate_page_level(model, metadata_file, global_label_map, DEVICE)
            print(f"Epoch {epoch} Test | 页面级准确率: {test_page_acc:.4f}")

            if test_page_acc > best_page_acc:
                best_page_acc = test_page_acc
                no_improve_count = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_page_acc,
                }, best_model_path)
                print(f"🎉 新最佳模型保存 (页面 Acc: {best_page_acc:.4f})")
            else:
                no_improve_count += 1
                print(f"⚠️ 验证未提升 ({no_improve_count}/{PATIENCE})")

        # 保存最新完整检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_page_acc,
        }, latest_model_path)

        # 写日志
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{epoch_loss:.4f}",
                f"{epoch_acc:.4f}",
                f"{test_page_acc:.4f}",
            ])

        # ========== 早停 ==========
        if no_improve_count >= PATIENCE:
            print(f"[EARLY STOP] 验证准确率连续 {PATIENCE} 次未提升，停止训练。")
            break

        print("-" * 50)