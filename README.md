# Baseline Models

本目录包含多个对比模型的实现，用于图文检索和多模态对齐实验。

## 📁 目录结构

```
baseline/
├── common/              # 共享工具和配置
│   ├── __init__.py
│   ├── config.py        # 基础配置类
│   ├── metrics.py       # 评估指标工具
│   ├── dataset.py       # 数据集基类
│   ├── data.py          # 数据处理工具
│   └── training.py      # 训练和验证函数
│
├── models/              # 各个模型实现
│   ├── __init__.py
│   ├── clip/           # CLIP模型
│   │   ├── __init__.py
│   │   ├── config.py   # CLIP配置
│   │   ├── model.py    # CLIP模型定义
│   │   ├── train.py    # CLIP训练脚本
│   │   └── evaluate.py # CLIP评估脚本
│   └── template/       # 新模型模板
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── train.py
│       └── evaluate.py
│
├── train.py            # 统一训练入口
├── evaluate.py         # 统一评估入口
└── README.md          # 本文件
```

## 🚀 使用方法

### 训练模型

```bash
# 训练CLIP模型
python train.py --model clip --model-path best.pt
```

### 评估模型

```bash
# 评估CLIP模型
python evaluate.py --model clip --model-path best.pt --query "a group of people dancing in a party"
```

## 📦 添加新模型

要添加新的baseline模型，请按照以下步骤：

### 方法1: 使用模板（推荐）

我们提供了完整的模板文件在 `models/template/` 目录下：

```bash
# 1. 复制模板目录
cp -r models/template models/your_model

# 2. 根据需要修改文件
# - config.py: 设置模型配置
# - model.py: 实现模型架构
# - train.py: 自定义训练逻辑
# - evaluate.py: 自定义评估逻辑
```

### 方法2: 手动创建

在 `models/your_model/` 目录下创建：

- `__init__.py` - 模块导出
- `config.py` - 模型配置（继承自 `BaseConfig`）
- `model.py` - 模型定义
- `train.py` - 训练脚本
- `evaluate.py` - 评估脚本

### 更新主入口

1. 在 `train.py` 中添加新模型：
```python
from models.your_model import train_your_model

# 在main()函数的if语句中添加：
elif args.model == 'your_model':
    train_your_model.train(model_path=args.model_path)
```

2. 在 `evaluate.py` 中添加新模型：
```python
from models.your_model import evaluate_your_model

# 在main()函数的if语句中添加：
elif args.model == 'your_model':
    evaluate_your_model.evaluate(model_path=args.model_path, query=args.query)
```

3. 在 `models/__init__.py` 中导入：
```python
from . import your_model
```

## 🔧 配置说明

所有模型都继承自 `BaseConfig`，包含以下配置：

- **数据路径**: `image_path`, `captions_path`
- **训练参数**: `batch_size`, `epochs`, `device`
- **优化器参数**: `head_lr`, `image_encoder_lr`, `text_encoder_lr`
- **模型参数**: `size`, `image_embedding`, `text_embedding`

特定模型可以在自己的配置类中覆盖这些值。

## 📊 当前支持的模型

- ✅ **CLIP**: Contrastive Language-Image Pre-training
- 🚧 更多模型即将推出...

## 📝 依赖项

- PyTorch
- transformers
- timm
- albumentations
- sklearn
- pandas
- opencv-python
- matplotlib

## 🔍 数据集

默认使用Flickr30k数据集。可以在 `common/config.py` 中修改数据路径。
