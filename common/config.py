"""
基础配置类
"""
import torch
import os


class BaseConfig:
    """所有模型的基类配置"""

    # ============ 项目根目录 ============
    project_root = "D:\\code\\causality\\FrontdoorCausalChain"

    def __init__(self):
        """初始化配置"""
        # ============ 数据集选择 ============
        # 可选: 'flickr30k', 'mm_celeba_hq', 'mscoco_15k'
        self.dataset_name = 'flickr30k'

        # ============ 文本模型路径 ============
        self.text_model_path = os.path.join(self.project_root, 'PreTrainedModels', 'distilbert_base_uncased')

    # ============ 数据集路径配置 ============
    @property
    def dataset_path(self):
        """获取当前选择的数据集路径"""
        dataset_paths = {
            'flickr30k': os.path.join(self.project_root, 'data', 'flickr30k'),
            'mm_celeba_hq': os.path.join(self.project_root, 'data', 'MM-CELEBA-HQ'),
            'mscoco_15k': os.path.join(self.project_root, 'data', 'mscoco_15k')
        }
        return dataset_paths.get(self.dataset_name,
                                 os.path.join(self.project_root, 'data', 'flickr30k'))

    @property
    def image_path(self):
        """获取当前数据集的图片路径"""
        image_paths = {
            'flickr30k': os.path.join(self.dataset_path, 'flickr30k_images'),
            'mm_celeba_hq': os.path.join(self.dataset_path, 'images'),
            'mscoco_15k': self.dataset_path  # mscoco 使用 arrow 格式，不需要单独的图片路径
        }
        return image_paths.get(self.dataset_name, self.dataset_path)

    @property
    def captions_path(self):
        """获取当前数据集的文本描述路径"""
        caption_paths = {
            'flickr30k': self.dataset_path,
            'mm_celeba_hq': os.path.join(self.dataset_path, 'text'),
            'mscoco_15k': self.dataset_path
        }
        return caption_paths.get(self.dataset_name, self.dataset_path)

    @property
    def train_path(self):
        """获取训练集路径（用于 mscoco_15k）"""
        if self.dataset_name == 'mscoco_15k':
            return os.path.join(self.dataset_path, 'mscoco_15k_train')
        return self.dataset_path

    @property
    def valid_path(self):
        """获取验证集路径（用于 mscoco_15k）"""
        if self.dataset_name == 'mscoco_15k':
            return os.path.join(self.dataset_path, 'mscoco_15k_test')
        return self.dataset_path

    # ============ 数据集特定配置说明 ============
    # flickr30k: captions.txt 文件包含所有图片的描述
    # MM-CELEBA-HQ: 每张图片对应一个 txt 文件（如 0.jpg 对应 0.txt），每行一条描述
    # mscoco_15k: 使用 .arrow 格式，分为训练集和验证集

    # ============ 训练参数 ============
    batch_size = 32
    num_workers = 4
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ 优化器参数 ============
    head_lr = 2 * 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8

    # ============ 图像参数 ============
    size = 224
    image_embedding = 2048

    # ============ 文本参数 ============
    text_embedding = 768
    max_length = 200
    text_tokenizer = "distilbert-base-uncased"

    # ============ 通用参数 ============
    pretrained = True
    trainable = True
    temperature = 1.0
    projection_dim = 256
    dropout = 0.1

    # ============ 调试模式 ============
    debug = False
