"""
数据集基类和工具
"""
import cv2
import torch
import os
from torch.utils.data import Dataset
import albumentations as A
from .config import BaseConfig as CFG


class BaseDataset(Dataset):
    """
    基础数据集类 - 用于加载图像-文本对
    支持从文件系统加载图像
    """

    def __init__(self, image_filenames, captions, ids, tokenizer, transforms, image_path=None):
        """
        Args:
            image_filenames: 图像文件名列表
            captions: 对应的文本描述列表
            ids: 样本ID列表
            tokenizer: 文本 tokenizer
            transforms: 图像变换
            image_path: 图像路径（可选，默认使用 CFG.image_path）
        """
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.ids = ids
        self.image_path = image_path or CFG.image_path
        self.encoded_captions = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        # 获取编码的文本
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # 读取和处理图像
        image_path = os.path.join(self.image_path, self.image_filenames[idx])
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        item['id'] = torch.tensor(self.ids[idx], dtype=torch.long)

        return item

    def __len__(self):
        return len(self.captions)


class ArrowDataset(Dataset):
    """
    Arrow 格式数据集类
    用于 MSCOCO-15k 等使用 .arrow 格式的数据集
    图像数据已预编码并存储在 arrow 文件中
    """

    def __init__(self, image_data, captions, ids, tokenizer, transforms, image_path=None):
        """
        Args:
            image_data: 图像数据（numpy 数组或已解码的字节）
            captions: 对应的文本描述列表
            ids: 样本ID列表
            tokenizer: 文本 tokenizer
            transforms: 图像变换
            image_path: 图像路径（保留用于兼容性，但不使用）
        """
        self.image_data = image_data
        self.captions = list(captions)
        self.ids = ids
        self.encoded_captions = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        # 获取编码的文本
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # 获取图像数据
        img_data = self.image_data[idx]

        # 处理不同格式的图像数据
        if isinstance(img_data, bytes):
            # 如果是字节流，需要解码
            import numpy as np
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # 如果已经是 numpy 数组
            image = img_data

        if image is None:
            raise ValueError(f"无法解码图像数据，索引: {idx}")

        # 确保 RGB 格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用变换
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        item['id'] = torch.tensor(self.ids[idx], dtype=torch.long)

        return item

    def __len__(self):
        return len(self.captions)


def get_transforms(mode="train", size=224):
    """
    获取图像变换

    Args:
        mode: "train" 或 "valid"/"test"
        size: 图像尺寸

    Returns:
        albumentations 变换组合
    """
    return A.Compose([
        A.Resize(size, size, p=1.0),
        A.Normalize(max_pixel_value=255.0, p=1.0),
    ])
