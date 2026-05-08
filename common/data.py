"""
数据处理相关工具
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from .config import BaseConfig as CFG
from .dataset import BaseDataset, get_transforms


def make_train_valid_dfs(test_size=0.2, random_state=42):
    """
    创建训练和验证数据集

    Args:
        test_size: 验证集比例
        random_state: 随机种子

    Returns:
        train_dataframe: 训练数据DataFrame
        valid_dataframe: 验证数据DataFrame
    """
    dataframe = pd.read_csv(f"{CFG.captions_path}\captions.txt")
    max_id = dataframe.shape[0] if not CFG.debug else 100

    train_dataframe, valid_dataframe = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_dataframe = train_dataframe.reset_index(names="original_index")
    valid_dataframe = valid_dataframe.reset_index(names="original_index")

    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode, dataset_class=BaseDataset):
    """
    构建数据加载器

    Args:
        dataframe: 包含图像和文本数据的DataFrame
        tokenizer: 文本tokenizer
        mode: "train" 或 "valid"/"test"
        dataset_class: 数据集类（允许自定义）

    Returns:
        dataloader: PyTorch数据加载器
    """
    transforms = get_transforms(mode=mode)
    dataset = dataset_class(
        dataframe["image_name"].values,
        dataframe["comment"].values,
        dataframe["original_index"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader
