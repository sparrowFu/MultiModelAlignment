# FrontdoorCausalChain

基于前门准则的多模态因果链学习项目，支持图文检索、因果推断等多种任务。

## 项目概述

本项目实现了多种多模态模型，包括：

- **CLIP 模型**: 基于对比学习的图文检索模型
- **FrontDoor 因果链模型**: 基于前门准则的因果推断模型

## 目录结构

```
FrontdoorCausalChain/
├── common/                    # 共享工具和配置
│   ├── __init__.py
│   ├── config.py             # 基础配置类
│   ├── data.py               # 数据处理工具
│   ├── dataset.py            # 数据集类
│   ├── dataset_loaders.py    # 多数据集加载器
│   ├── metrics.py            # 评估指标工具
│   └── training.py           # 训练和验证函数
│
├── models/                    # 模型实现目录
│   ├── __init__.py
│   ├── clip/                 # CLIP 模型
│   │   ├── __init__.py
│   │   ├── config.py         # CLIP 配置
│   │   ├── model.py          # CLIP 模型定义
│   │   ├── train.py          # CLIP 训练脚本
│   │   └── evaluate.py       # CLIP 评估脚本
│   ├── frontdoor/            # FrontDoor 因果链模型
│   │   ├── __init__.py
│   │   ├── config.py         # FrontDoor 配置
│   │   ├── model.py          # FrontDoor 模型定义
│   │   ├── loss.py           # FrontDoor 损失函数
│   │   ├── train.py          # FrontDoor 训练脚本
│   │   └── evaluate.py       # FrontDoor 评估脚本
│   └── template/             # 新模型模板
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── train.py
│       └── evaluate.py
│
├── data/                  # 数据集目录
│   ├── flickr30k/            # Flickr30k 数据集
│   ├── MM-CELEBA-HQ/         # MM-CELEBA-HQ 数据集
│   └── mscoco_15k/           # MSCOCO-15k 数据集
│
├── PreTrainedModels/          # 预训练模型
│   ├── distilbert_base_uncased/
│   └── resnet50/
│
├── train.py                   # 统一训练入口
├── train_causal_chain.py     # FrontDoor 训练脚本
├── evaluate.py                # 统一评估入口
└── results/                   # 训练结果输出
```

## 支持的数据集

项目支持三种数据集，可通过命令行参数选择：

| 数据集 | 说明 | 格式 |
|--------|------|------|
| `flickr30k` | Flickr30k 图文数据集 | captions.txt |
| `mm_celeba_hq` | MM-CELEBA-HQ 人脸属性数据集 | 图片-txt 对应 |
| `mscoco_15k` | MSCOCO-15k 图文数据集 | .arrow 格式 |

## 快速开始

### 安装依赖

```bash
pip install torch transformers timm albumentations sklearn pandas opencv-python matplotlib tqdm pyarrow
```

### 训练模型

```bash
# 使用默认数据集 (flickr30k) 训练 FrontDoor 模型
python train_causal_chain.py

# 选择其他数据集
python train_causal_chain.py --dataset mm_celeba_hq
python train_causal_chain.py --dataset mscoco_15k

# 自定义训练参数
python train_causal_chain.py --dataset flickr30k --batch-size 64 --epochs 10 --lr 1e-4

# 启用调试模式（使用少量数据）
python train_causal_chain.py --debug
```

### 评估模型

```bash
# 评估 CLIP 模型
python evaluate.py --model clip --model-path best.pt --query "a beautiful sunset"

# 评估 FrontDoor 模型
cd models/frontdoor
python evaluate.py
```

## 配置说明

### 数据集配置

所有模型都继承自 `BaseConfig`，可以通过以下方式配置数据集：

```python
from common.config import BaseConfig

config = BaseConfig()
config.dataset_name = 'flickr30k'  # 或 'mm_celeba_hq', 'mscoco_15k'

# 数据集路径会自动根据 dataset_name 设置
# config.dataset_path  # 数据集根目录
# config.image_path    # 图片目录
# config.captions_path # 文本描述目录
```

### 模型配置

每个模型都有独立的配置类，继承自 `BaseConfig`：

```python
from models.frontdoor.config import FrontDoorConfig

config = FrontDoorConfig()
# 覆盖默认配置
config.shared_dim = 512
config.batch_size = 64
config.epochs = 20
```

## 模型说明

### CLIP 模型

基于对比学习的图文检索模型，使用 ResNet50 和 DistilBERT 作为编码器。

- **输入**: 图像 + 文本
- **输出**: 图文相似度
- **损失函数**: 对比学习损失

### FrontDoor 因果链模型

基于前门准则的因果推断模型，将特征分解为 shared 和 private 部分。

- **输入**: 图像 + 文本
- **输出**: 因果效应值
- **损失函数**: 组合损失（对齐、正交、对比、重建）

**模型特点**:
- Shared/Private 特征分解
- 共享语义融合
- 因果效应估计
- 多重损失优化

## 添加新模型

### 使用模板（推荐）

```bash
# 1. 复制模板目录
cp -r models/template models/your_model

# 2. 修改配置
# models/your_model/config.py
class YourModelConfig(BaseConfig):
    model_name = 'your_model'

# 3. 实现模型
# models/your_model/model.py
class YourModel(nn.Module):
    def forward(self, batch):
        # 实现你的模型逻辑
        return loss

# 4. 更新 models/__init__.py
from . import your_model
```

## 依赖项

- PyTorch >= 1.10
- transformers >= 4.20
- timm >= 0.6
- albumentations >= 1.0
- scikit-learn
- pandas
- opencv-python
- matplotlib
- tqdm
- pyarrow

## 文档

- [README.md](README.md) - 项目说明（本文件）
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [ARCHITECTURE.md](ARCHITECTURE.md) - 架构详细说明
- [INDEX.md](INDEX.md) - 文件索引

## 许可证

本项目仅用于学术研究和教育目的。
