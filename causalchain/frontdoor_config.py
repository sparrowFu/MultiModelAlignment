"""
FrontDoorCausalChain 训练配置
"""
import torch
from baseline.common.config import BaseConfig


class FrontDoorConfig(BaseConfig):
    """FrontDoor 因果链训练配置"""

    # 文本模型路径（确保末尾无分隔符）
    text_model_path = r'D:\code\causality\models\distilbert_base_uncased'
    
    # 模型架构参数
    shared_dim = 256              # 共享语义维度
    private_ratio = 0.3           # private特征比例

    # 损失函数权重
    lambda_alignment = 1.0        # shared特征对齐损失权重
    lambda_orthogonal = 0.1       # 正交损失权重
    lambda_contrastive = 1.0      # 对比学习损失权重
    lambda_reconstruction = 0.5   # 重建损失权重

    # 训练参数
    epochs = 10
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-4

    # 温度参数（用于对比学习）
    temperature = 0.07

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 保存路径
    model_save_path = "d:\\code\\causality\\models\\frontdoor\\best_model.pt"
    checkpoint_path = "d:\\code\\causality\\models\\frontdoor\\checkpoint.pt"

    # 日志
    log_interval = 10             # 每10个batch打印一次
    save_interval = 1             # 每1个epoch保存一次
