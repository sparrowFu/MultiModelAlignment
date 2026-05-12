"""
数据处理相关工具
"""
import torch
from .config import BaseConfig as CFG
from .BaseDataset import BaseDataset, ArrowDataset, get_transforms
from .dataset_loaders import make_train_valid_dfs as load_data


def make_train_valid_dfs(test_size=0.2, random_state=42, config=None):
    """
    创建训练和验证数据集

    Args:
        test_size: 验证集比例
        random_state: 随机种子
        config: 配置对象（可选，默认使用 CFG）

    Returns:
        train_dataframe: 训练数据 DataFrame
        valid_dataframe: 验证数据 DataFrame
    """
    if config is None:
        config = CFG()

    return load_data(config=config, test_size=test_size, random_state=random_state)


def build_loaders(dataframe, tokenizer, mode, config=None, dataset_class=None):
    """
    构建数据加载器

    Args:
        dataframe: 包含图像和文本数据的 DataFrame
        tokenizer: 文本 tokenizer
        mode: "train" 或 "valid"/"test"
        config: 配置对象（可选，默认使用 CFG）
        dataset_class: 数据集类（可选，自动选择）

    Returns:
        dataloader: PyTorch 数据加载器
    """
    if config is None:
        config = CFG()

    transforms = get_transforms(mode=mode, size=config.size)

    # 根据数据集类型选择数据集类
    if dataset_class is None:
        if config.dataset_name == 'mscoco_15k':
            # mscoco_15k 使用 Arrow 数据集
            dataset_class = ArrowDataset
        else:
            # 其他数据集使用基础数据集
            dataset_class = BaseDataset

    dataset = dataset_class(
        dataframe["image"].values,
        dataframe["caption"].values,
        dataframe["original_index"].values,
        tokenizer=tokenizer,
        transforms=transforms,
        image_path=config.image_path,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader
