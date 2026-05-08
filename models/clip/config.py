"""
CLIP模型配置
"""
import torch
from common.config import BaseConfig
from pathlib import Path

class CLIPConfig(BaseConfig):
    """CLIP模型特定配置"""

    # 模型名称
    model_name = 'resnet50'
    text_model_path = 'D:\\code\\causality\\models\\distilbert_base_uncased'
    text_encoder_model = "distilbert-base-uncased"

    # 投影头参数
    num_projection_layers = 1

    @classmethod
    def to_dict(cls):
        """转换为字典格式"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
