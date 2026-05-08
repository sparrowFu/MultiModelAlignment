"""
FrontDoor 因果链训练脚本（独立运行版本）

运行方式:
    python -m baseline.train_causal_chain
    或
    python baseline/train_causal_chain.py
"""
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer

from baseline.causalchain.frontdoor_config import FrontDoorConfig
from baseline.causalchain.frontdoor_model import FrontDoorCausalModel, FrontDoorWithEncoders
from baseline.causalchain.frontdoor_loss import FrontDoorLoss
from baseline.common.data import make_train_valid_dfs, build_loaders
from baseline.models.clip.model import ImageEncoder, TextEncoder


def load_encoders(device):
    """加载预训练的图像和文本编码器"""
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder().to(device)

    # 加载预训练权重（如果存在）
    clip_model_path = "D:\\code\\causality\\baseline\\best.pt"
    if os.path.exists(clip_model_path):
        print(f"加载预训练CLIP模型: {clip_model_path}")
        checkpoint = torch.load(clip_model_path, map_location=device, weights_only=False)

        # CLIP模型保存的是完整模型状态，需要分别提取编码器
        image_encoder.load_state_dict({
            k.replace('image_encoder.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('image_encoder.')
        }, strict=False)
        text_encoder.load_state_dict({
            k.replace('text_encoder.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('text_encoder.')
        }, strict=False)

    # 设置为评估模式并冻结参数
    image_encoder.eval()
    text_encoder.eval()
    for param in image_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    return image_encoder, text_encoder


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(batch)

        with torch.no_grad():
            image_features = model.image_encoder(batch["image"])
            text_features = model.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

        losses = criterion(output, image_features, text_features, batch["id"])
        loss = losses['total_loss']

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def valid_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            output = model(batch)
            image_features = model.image_encoder(batch["image"])
            text_features = model.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            losses = criterion(output, image_features, text_features, batch["id"])
            total_loss += losses['total_loss'].item()

    return total_loss / len(dataloader)


def main():
    """主训练函数"""
    config = FrontDoorConfig()
    device = config.device

    print("=" * 60)
    print("FrontDoor 因果链训练")
    print("=" * 60)

    # 准备数据
    print("\n准备数据...")
    train_df, valid_df = make_train_valid_dfs(test_size=0.2, random_state=42)
    print(f"训练样本: {len(train_df)}, 验证样本: {len(valid_df)}")

    tokenizer = DistilBertTokenizer.from_pretrained(
        config.text_model_path,
        local_files_only=True
    )
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    # 加载编码器
    print("\n加载预训练编码器...")
    image_encoder, text_encoder = load_encoders(device)

    # 创建模型
    causal_model = FrontDoorCausalModel(
        image_feat_dim=config.image_embedding,
        text_feat_dim=config.text_embedding,
        shared_dim=config.shared_dim,
        config=config
    ).to(device)
    model = FrontDoorWithEncoders(image_encoder, text_encoder, causal_model)

    criterion = FrontDoorLoss(config)
    optimizer = torch.optim.AdamW(causal_model.parameters(), lr=config.lr)

    # 训练
    best_loss = float('inf')
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        val_loss = valid_epoch(model, valid_loader, criterion, device)

        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            torch.save(causal_model.state_dict(), config.model_save_path)
            print(f"✅ 保存最佳模型!")

    print(f"\n训练完成! 最佳损失: {best_loss:.4f}")


if __name__ == "__main__":
    main()
