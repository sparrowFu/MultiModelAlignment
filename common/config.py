"""
基础配置类
"""
import torch

class BaseConfig:
    """所有模型的基类配置"""

    # 数据路径
    image_path = "D:\\code\\causality\\datasets\\flickr30k\\flickr30k_images\\"
    captions_path = "D:\\code\\causality\\datasets\\flickr30k"

    # 训练参数
    batch_size = 32
    num_workers = 4
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优化器参数
    head_lr = 2 * 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8

    # 图像参数
    size = 224
    image_embedding = 2048

    # 文本参数
    text_embedding = 768
    max_length = 200
    text_tokenizer = "distilbert-base-uncased"
    text_model_path = 'D:\\code\\causality\\models\\distilbert_base_uncased'

    # 通用参数
    pretrained = True
    trainable = True
    temperature = 1.0
    projection_dim = 256
    dropout = 0.1

    # 调试模式
    debug = False
