# 文件索引

本文档提供了项目中所有文件的索引和说明。

## 根目录文件

### 文档文件

| 文件 | 说明 |
|------|------|
| [README.md](README.md) | 项目主文档 |
| [QUICKSTART.md](QUICKSTART.md) | 快速开始指南 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构详细说明 |
| [INDEX.md](INDEX.md) | 文件索引（本文件） |

### 入口脚本

| 文件 | 说明 |
|------|------|
| [train.py](train.py) | 统一训练入口 |
| [train_causal_chain.py](train_causal_chain.py) | FrontDoor 模型训练脚本 |
| [evaluate.py](evaluate.py) | 统一评估入口 |

### 工具脚本

| 文件 | 说明 |
|------|------|
| [check_structure.py](check_structure.py) | 项目结构检查 |
| [test_imports.py](test_imports.py) | 导入测试 |

## common/ 目录

共享工具和基础类。

| 文件 | 说明 |
|------|------|
| [__init__.py](common/__init__.py) | 模块导出 |
| [config.py](common/config.py) | 基础配置类 BaseConfig |
| [data.py](common/data.py) | 数据处理工具 |
| [dataset.py](common/dataset.py) | 数据集类 BaseDataset, ArrowDataset |
| [dataset_loaders.py](common/dataset_loaders.py) | 多数据集加载器 |
| [metrics.py](common/metrics.py) | 评估指标工具 AvgMeter, get_lr |
| [training.py](common/training.py) | 训练和验证函数 |

### 核心类说明

- **BaseConfig**: 所有模型的基类配置，包含数据集路径配置
- **BaseDataset**: 常规图片数据集类
- **ArrowDataset**: .arrow 格式数据集类
- **DatasetLoader**: 数据集加载器基类
- **Flickr30kLoader**: Flickr30k 数据集加载器
- **MMCelebaHQLoader**: MM-CELEBA-HQ 数据集加载器
- **MSCOCO15kLoader**: MSCOCO-15k 数据集加载器

## models/ 目录

模型实现目录。

### models/clip/

CLIP 模型实现。

| 文件 | 说明 |
|------|------|
| [__init__.py](models/clip/__init__.py) | 模块导出 |
| [config.py](models/clip/config.py) | CLIP 配置类 |
| [model.py](models/clip/model.py) | CLIP 模型定义 |
| [train.py](models/clip/train.py) | CLIP 训练脚本 |
| [evaluate.py](models/clip/evaluate.py) | CLIP 评估脚本 |

**核心类**:
- `ImageEncoder`: 图像编码器 (ResNet50)
- `TextEncoder`: 文本编码器 (DistilBERT)
- `ProjectionHead`: 投影头
- `CLIPModel`: CLIP 主模型

### models/frontdoor/

FrontDoor 因果链模型实现。

| 文件 | 说明 |
|------|------|
| [__init__.py](models/frontdoor/__init__.py) | 模块导出 |
| [config.py](models/frontdoor/config.py) | FrontDoor 配置类 |
| [model.py](models/frontdoor/model.py) | FrontDoor 模型定义 |
| [loss.py](models/frontdoor/loss.py) | FrontDoor 损失函数 |
| [train.py](models/frontdoor/train.py) | FrontDoor 训练脚本 |
| [evaluate.py](models/frontdoor/evaluate.py) | FrontDoor 评估脚本 |

**核心类**:
- `FrontDoorCausalModel`: 因果链模型
- `FrontDoorWithEncoders`: 包含编码器的完整模型
- `FrontDoorLoss`: 组合损失函数

**损失函数**:
- 对齐损失 (alignment_loss)
- 正交损失 (orthogonal_loss)
- 对比损失 (contrastive_loss)
- 重建损失 (reconstruction_loss)

### models/template/

新模型模板。

| 文件 | 说明 |
|------|------|
| [__init__.py](models/template/__init__.py) | 模块导出 |
| [config.py](models/template/config.py) | 模板配置类 |
| [model.py](models/template/model.py) | 模板模型定义 |
| [train.py](models/template/train.py) | 模板训练脚本 |
| [evaluate.py](models/template/evaluate.py) | 模板评估脚本 |

## data/ 目录

数据集存储目录。

### data/flickr30k/

Flickr30k 数据集。

```
flickr30k/
├── flickr30k_images/    # 图片目录
│   ├── 1000092795.jpg
│   └── ...
└── captions.txt         # 图片描述文件
```

### data/MM-CELEBA-HQ/

MM-CELEBA-HQ 数据集。

```
MM-CELEBA-HQ/
├── images/              # 图片目录
│   ├── 0.jpg
│   └── ...
└── text/                # 文本描述目录
    ├── 0.txt            # 每个文件包含10条描述
    └── ...
```

### data/mscoco_15k/

MSCOCO-15k 数据集。

```
mscoco_15k/
├── mscoco_15k_train/    # 训练集
│   └── data.arrow       # Arrow 格式数据
└── mscoco_15k_test/     # 验证集
    └── data.arrow
```

## PreTrainedModels/ 目录

预训练模型存储目录。

```
PreTrainedModels/
├── distilbert_base_uncased/    # DistilBERT 模型
└── resnet50/                    # ResNet50 模型
```

## results/ 目录

训练结果输出目录（训练时自动创建）。

```
results/
└── frontdoormodel/
    ├── best_model.pt           # 最佳模型权重
    └── checkpoint.pt           # 训练检查点
```

## 文件统计

| 类型 | 数量 |
|------|------|
| Python 文件 | 28 |
| Markdown 文档 | 4 |
| 模块 | 3 (clip, frontdoor, template) |
| 支持的数据集 | 3 |

## 依赖关系图

```
train_causal_chain.py
    ├── models.frontdoor
    │   ├── common.config
    │   ├── common.data
    │   ├── common.dataset
    │   ├── models.clip (编码器)
    │   └── common.dataset_loaders
    └── common
        ├── config.py
        ├── data.py
        ├── dataset.py
        ├── dataset_loaders.py
        ├── metrics.py
        └── training.py
```

## 快速导航

### 添加新模型
1. 复制 `models/template/`
2. 修改配置和模型
3. 更新 `models/__init__.py`

### 添加新数据集
1. 在 `common/dataset_loaders.py` 创建新的加载器
2. 在 `get_dataset_loader` 中注册
3. 在 `BaseConfig` 中添加路径配置（如需要）

### 修改训练配置
1. 修改 `models/*/config.py` 中的模型配置
2. 或修改 `common/config.py` 中的基础配置
