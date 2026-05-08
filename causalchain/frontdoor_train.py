"""
FrontDoorCausalChain 训练脚本
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from transformers import DistilBertTokenizer

from .frontdoor_config import FrontDoorConfig
from .frontdoor_model import FrontDoorCausalModel, FrontDoorWithEncoders
from .frontdoor_loss import FrontDoorLoss
from baseline.common.data import make_train_valid_dfs, build_loaders
from baseline.common.dataset import BaseDataset, get_transforms
from baseline.models.clip.model import ImageEncoder, TextEncoder


def load_encoders(device):
    """
    加载预训练的图像和文本编码器

    Args:
        device: torch设备

    Returns:
        image_encoder, text_encoder: 预训练编码器
    """
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder().to(device)

    # 加载预训练权重（如果存在）
    clip_model_path = "D:\\code\\causality\\baseline\\best.pt"
    if os.path.exists(clip_model_path):
        print(f"加载预训练CLIP模型: {clip_model_path}")
        checkpoint = torch.load(clip_model_path, map_location=device)

        # CLIP模型保存的是完整模型状态，需要分别提取编码器
        # 假设checkpoint是CLIPModel的state_dict
        if 'image_encoder.model' in checkpoint:
            image_encoder.load_state_dict({
                k.replace('image_encoder.', ''): v
                for k, v in checkpoint.items()
                if k.startswith('image_encoder.')
            })
        if 'text_encoder.model' in checkpoint:
            text_encoder.load_state_dict({
                k.replace('text_encoder.', ''): v
                for k, v in checkpoint.items()
                if k.startswith('text_encoder.')
            })

    # 设置为评估模式
    image_encoder.eval()
    text_encoder.eval()

    # 冻结编码器参数（不训练预训练编码器）
    for param in image_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    return image_encoder, text_encoder


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_losses = {
        'alignment': 0,
        'orthogonal': 0,
        'contrastive': 0,
        'reconstruction': 0
    }

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 前向传播
        optimizer.zero_grad()
        output = model(batch)

        # 获取编码特征（用于重建损失）
        with torch.no_grad():
            image_features = model.image_encoder(batch["image"])
            text_features = model.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

        # 计算损失
        losses = criterion(
            output, image_features, text_features,
            batch["id"]
        )

        loss = losses['total_loss']

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        all_losses['alignment'] += losses['alignment_loss'].item()
        all_losses['orthogonal'] += losses['orthogonal_loss'].item()
        all_losses['contrastive'] += losses['contrastive_loss'].item()
        all_losses['reconstruction'] += losses['reconstruction_loss'].item()

        # 更新进度条
        if (batch_idx + 1) % config.log_interval == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'align': f"{losses['alignment_loss'].item():.4f}",
                'ortho': f"{losses['orthogonal_loss'].item():.4f}"
            })

    # 计算平均损失
    num_batches = len(dataloader)
    for k in all_losses:
        all_losses[k] /= num_batches

    return total_loss / num_batches, all_losses


def valid_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_metrics = {
        'shared_similarity': 0,
        'image_orthogonality': 0,
        'text_orthogonality': 0,
        'avg_orthogonality': 0
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # 移动到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 前向传播
            output = model(batch)

            # 获取编码特征
            image_features = model.image_encoder(batch["image"])
            text_features = model.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            # 计算损失
            losses = criterion(
                output, image_features, text_features,
                batch["id"]
            )
            total_loss += losses['total_loss'].item()

            # 获取指标
            metrics = criterion.get_metrics(output)
            for k, v in metrics.items():
                all_metrics[k] += v

    # 计算平均值
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for k in all_metrics:
        all_metrics[k] /= num_batches

    return avg_loss, all_metrics


def save_checkpoint(model, optimizer, epoch, best_loss, config, path):
    """保存训练checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.causal_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'config': config.__dict__
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """加载训练checkpoint"""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.causal_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['best_loss']
    return 0, float('inf')


def train(config=None, resume=True):
    """
    训练 FrontDoor 因果链模型

    Args:
        config: FrontDoorConfig 配置对象
        resume: 是否从checkpoint恢复训练
    """
    config = config or FrontDoorConfig()
    device = config.device

    print("=" * 60)
    print("FrontDoor 因果链训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"共享语义维度: {config.shared_dim}")
    print(f"Batch大小: {config.batch_size}")
    print(f"Epochs: {config.epochs}")

    # 准备数据
    print("\n准备数据...")
    train_df, valid_df = make_train_valid_dfs(test_size=0.2, random_state=42)
    print(f"训练样本数: {len(train_df)}")
    print(f"验证样本数: {len(valid_df)}")

    # 去除路径末尾的分隔符，避免 tokenizer 加载错误
    model_path = str(config.text_model_path).rstrip(os.sep)
    tokenizer = DistilBertTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    # 加载预训练编码器
    print("\n加载预训练编码器...")
    image_encoder, text_encoder = load_encoders(device)

    # 创建因果模型
    print("\n初始化因果模型...")
    causal_model = FrontDoorCausalModel(
        image_feat_dim=config.image_embedding,
        text_feat_dim=config.text_embedding,
        shared_dim=config.shared_dim,
        private_ratio=config.private_ratio,
        config=config
    ).to(device)

    # 组合模型
    model = FrontDoorWithEncoders(image_encoder, text_encoder, causal_model)

    # 损失函数
    criterion = FrontDoorLoss(config)

    # 优化器（只训练causal_model）
    optimizer = torch.optim.AdamW(
        causal_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')

    if resume:
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, config.checkpoint_path, device
        )
        if start_epoch > 0:
            print(f"从 epoch {start_epoch} 恢复训练，最佳损失: {best_loss:.4f}")

    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 60)

        # 训练
        train_loss, train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )

        print(f"训练损失: {train_loss:.4f}")
        print(f"  - 对齐损失: {train_losses['alignment']:.4f}")
        print(f"  - 正交损失: {train_losses['orthogonal']:.4f}")
        print(f"  - 对比损失: {train_losses['contrastive']:.4f}")
        print(f"  - 重建损失: {train_losses['reconstruction']:.4f}")

        # 验证
        val_loss, val_metrics = valid_epoch(
            model, valid_loader, criterion, device
        )

        print(f"\n验证损失: {val_loss:.4f}")
        print(f"  - Shared相似度: {val_metrics['shared_similarity']:.4f}")
        print(f"  - 图像正交性: {val_metrics['image_orthogonality']:.4f}")
        print(f"  - 文本正交性: {val_metrics['text_orthogonality']:.4f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            torch.save(causal_model.state_dict(), config.model_save_path)
            print(f"✅ 保存最佳模型! 损失: {best_loss:.4f}")

        # 保存checkpoint
        save_checkpoint(
            model, optimizer, epoch + 1, best_loss,
            config, config.checkpoint_path
        )

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证损失: {best_loss:.4f}")
    print(f"模型已保存至: {config.model_save_path}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    # 创建配置
    config = FrontDoorConfig()

    # 训练模型
    model = train(config, resume=True)
