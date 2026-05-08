"""
数据集基类和工具
"""
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from .config import BaseConfig as CFG


class BaseDataset(Dataset):
    """
    基础数据集类 - 用于加载图像-文本对
    """

    def __init__(self, image_filenames, captions, ids, tokenizer, transforms):
        """
        Args:
            image_filenames: 图像文件名列表
            captions: 对应的文本描述列表
            ids: 样本ID列表
            tokenizer: 文本tokenizer
            transforms: 图像变换
        """
        self.image_filenames = image_filenames
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

        # 读取和处理图像
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        albumentations变换组合
    """
    return A.Compose([
        A.Resize(size, size, always_apply=True),
        A.Normalize(max_pixel_value=255.0, always_apply=True),
    ])
