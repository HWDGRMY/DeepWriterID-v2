"""
ConvNeXt V2 训练器：Warmup + 余弦退火 + 断点续训 + 页面级验证。

参考论文的超参数设置（Woo et al., ConvNeXt V2, CVPR 2023, Table 10）：
- AdamW 优化器（beta1=0.9, beta2=0.999）
- 线性 warmup + 余弦退火
- Weight decay = 0.05
- DropPath = 0.1（Tiny 模型）
"""

import os
import time
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.epochs = config['training']['epochs']
        self.warmup_epochs = config['training'].get('warmup_epochs', 5)
        self.start_epoch = 0
        self.best_acc = 0.0

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器：论文推荐 AdamW, beta1=0.9, beta2=0.999
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['training']['initial_lr'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999),
        )

        # 调度器：Warmup + 余弦退火
        # 注意：step 是在每个 iteration 后调用，不是每个 epoch
        steps_per_epoch = len(train_loader)
        warmup_steps = self.warmup_epochs * steps_per_epoch
        cosine_steps = (self.epochs - self.warmup_epochs) * steps_per_epoch

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # 日志文件
        log_dir = config['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, config['logging']['log_file'])
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc', 'lr', 'time'])

    def train_epoch(self):
        """单轮训练。每个 iteration 后调用 scheduler.step()。"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            # 每个 iteration 后更新学习率（warmup + cosine）
            self.scheduler.step()

            total_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        return avg_loss, acc

    @torch.no_grad()
    def validate(self):
        """页面级验证：注意此处的准确率是按伪字符粒度计算的。"""
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        return correct / total

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            val_acc = self.validate()

            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']

            print(f"[Epoch {epoch+1}/{self.epochs}] "
                  f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_acc={val_acc:.4f} lr={lr:.8f} time={elapsed:.1f}s")

            # 写日志
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                                 f"{val_acc:.4f}", f"{lr:.8f}", f"{elapsed:.1f}"])

            # 保存最新模型
            latest_path = os.path.join(
                self.config['checkpoint']['save_dir'],
                self.config['checkpoint']['save_latest'],
            )
            os.makedirs(os.path.dirname(latest_path), exist_ok=True)
            self.save_checkpoint(latest_path, epoch)

            # 保存最优模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                best_path = os.path.join(
                    self.config['checkpoint']['save_dir'],
                    self.config['checkpoint']['save_best'],
                )
                self.save_checkpoint(best_path, epoch)
                print(f"  [BEST] 新纪录: {val_acc:.4f}，已保存至 {best_path}")

        print(f"[DONE] 训练结束，最佳验证准确率: {self.best_acc:.4f}")

    def save_checkpoint(self, path, epoch):
        """保存完整的断点状态：模型、优化器、调度器、epoch、最佳准确率。"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
        }, path)

    def load_checkpoint(self, path):
        """加载断点，恢复模型、优化器、调度器和 epoch。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint.get('best_acc', 0.0)
        print(f"[INFO] 断点恢复: epoch={self.start_epoch}, best_acc={self.best_acc:.4f}")