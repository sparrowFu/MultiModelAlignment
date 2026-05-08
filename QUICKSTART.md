# 快速开始指南

## 安装依赖

```bash
pip install torch transformers timm albumentations sklearn pandas opencv-python matplotlib tqdm
```

## 数据准备

1. 下载Flickr30k数据集
2. 解压到 `../datasets/flickr30k/` 目录
3. 确保目录结构如下：
```
../datasets/flickr30k/
├── flickr30k_images/
│   └── *.jpg
└── captions.txt
```

## 训练模型

### 方式1: 使用统一入口

```bash
# 训练CLIP模型
python train.py --model clip --model-path best.pt
```

### 方式2: 直接运行模型脚本

```bash
# 训练CLIP模型
cd models/clip
python train.py
```

## 评估模型

### 方式1: 使用统一入口

```bash
# 评估CLIP模型
python evaluate.py --model clip --model-path best.pt --query "a dog running in the park"
```

### 方式2: 直接运行模型脚本

```bash
# 评估CLIP模型
cd models/clip
python evaluate.py
```

## 配置修改

所有配置都可以在相应的 `config.py` 文件中修改：

### 修改全局配置

编辑 `common/config.py`:

```python
class BaseConfig:
    batch_size = 32  # 修改批大小
    epochs = 10      # 修改训练轮数
    # ... 其他配置
```

### 修改特定模型配置

编辑 `models/clip/config.py`:

```python
class CLIPConfig(BaseConfig):
    model_name = 'resnet50'  # 修改图像编码器
    # ... 其他CLIP特定配置
```

## 添加新模型

### 使用模板（推荐）

```bash
# 1. 复制模板
cp -r models/template models/my_model

# 2. 修改配置
vim models/my_model/config.py

# 3. 实现模型
vim models/my_model/model.py

# 4. 测试训练
cd models/my_model
python train.py

# 5. 更新主入口
vim ../train.py
vim ../evaluate.py
```

## 常见问题

### Q: 如何修改数据路径？

A: 在 `common/config.py` 中修改 `image_path` 和 `captions_path`。

### Q: 如何使用不同的backbone？

A: 在对应模型的 `config.py` 中修改 `model_name`。

### Q: 如何调整学习率？

A: 在对应模型的 `config.py` 中修改 `image_encoder_lr`、`text_encoder_lr` 等参数。

### Q: 训练时显存不足怎么办？

A: 减小 `batch_size` 或使用梯度累积。

## 进阶用法

### 自定义数据集

继承 `BaseDataset` 并实现自己的数据集类：

```python
from common.dataset import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, ...):
        super().__init__(...)
        # 你的自定义逻辑
```

### 自定义损失函数

在模型的 `forward` 方法中实现自定义损失计算。

### 多GPU训练

使用PyTorch的 `DistributedDataParallel` 或 `DataParallel`。

## 实验记录

建议使用实验跟踪工具（如wandb、tensorboard）记录实验结果。

### 示例：集成wandb

```python
import wandb

# 在训练开始前
wandb.init(project="my-project", config=config.to_dict())

# 在训练循环中
wandb.log({"train_loss": train_loss.avg, "valid_loss": valid_loss.avg})
```

## 性能优化

### 数据加载优化

- 增加 `num_workers`
- 使用 `pin_memory=True`

### 训练速度优化

- 使用混合精度训练 (AMP)
- 梯度累积
- 学习率warmup

### 模型优化

- 模型蒸馏
- 知识蒸馏
- 模型剪枝

## 下一步

- 查看 [README.md](README.md) 了解详细架构
- 查看 `models/template/` 学习如何添加新模型
- 查看各个模型的实现源码
