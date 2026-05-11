# 快速开始指南

本指南将帮助你快速上手 FrontdoorCausalChain 项目。

## 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU 加速)

## 安装步骤

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers timm albumentations scikit-learn pandas opencv-python matplotlib tqdm pyarrow
```

### 2. 准备数据集

项目支持三种数据集，选择一种准备：

#### Flickr30k (推荐新手)

```bash
# 下载并解压到 datasets/flickr30k/
# 目录结构:
# datasets/flickr30k/
# ├── flickr30k_images/
# │   ├── 1000092795.jpg
# │   └── ...
# └── captions.txt
```

#### MM-CELEBA-HQ

```bash
# 下载并解压到 datasets/MM-CELEBA-HQ/
# 目录结构:
# datasets/MM-CELEBA-HQ/
# ├── images/
# │   ├── 0.jpg
# │   └── ...
# └── text/
#     ├── 0.txt
#     └── ...
```

#### MSCOCO-15k

```bash
# 下载并解压到 datasets/mscoco_15k/
# 目录结构:
# datasets/mscoco_15k/
# ├── mscoco_15k_train/
# │   └── data.arrow
# └── mscoco_15k_test/
#     └── data.arrow
```

### 3. 准备预训练模型

预训练模型已包含在 `PreTrainedModels/` 目录中：

```
PreTrainedModels/
├── distilbert_base_uncased/
└── resnet50/
```

## 快速运行

### 训练 FrontDoor 模型

```bash
# 使用默认配置（flickr30k 数据集）
python train_causal_chain.py

# 使用其他数据集
python train_causal_chain.py --dataset mm_celeba_hq
python train_causal_chain.py --dataset mscoco_15k

# 调整训练参数
python train_causal_chain.py --dataset flickr30k --batch-size 64 --epochs 5

# 调试模式（快速验证代码）
python train_causal_chain.py --debug
```

### 评估模型

```bash
# 评估 FrontDoor 模型
cd models/frontdoor
python evaluate.py

# 评估 CLIP 模型
cd ..
python evaluate.py --model clip --model-path best.pt
```

## 命令行参数说明

### train_causal_chain.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集选择 (flickr30k/mm_celeba_hq/mscoco_15k) | flickr30k |
| `--batch-size` | 批大小 | 32 |
| `--epochs` | 训练轮数 | 2 |
| `--lr` | 学习率 | 1e-5 |
| `--debug` | 启用调试模式 | False |
| `--device` | 设备选择 (cuda/cpu) | auto |

示例：

```bash
# 完整参数示例
python train_causal_chain.py \
    --dataset flickr30k \
    --batch-size 64 \
    --epochs 10 \
    --lr 1e-4 \
    --device cuda
```

## 配置文件

### 修改默认配置

编辑 `models/frontdoor/config.py`：

```python
class FrontDoorConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # 数据集选择
        self.dataset_name = 'flickr30k'  # 或 'mm_celeba_hq', 'mscoco_15k'

        # 模型参数
        self.shared_dim = 256
        self.private_ratio = 0.3

        # 训练参数
        self.batch_size = 32
        self.epochs = 2
        self.lr = 1e-5

        # 损失权重
        self.lambda_alignment = 1.0
        self.lambda_orthogonal = 0.1
        self.lambda_contrastive = 1.0
        self.lambda_reconstruction = 0.5
```

### 修改基础配置

编辑 `common/config.py` 中的 `BaseConfig` 类，所有模型都会继承这些配置。

## 常见问题

### Q1: 如何切换数据集？

**A:** 使用 `--dataset` 参数：

```bash
python train_causal_chain.py --dataset mm_celeba_hq
```

或在配置文件中设置：

```python
config.dataset_name = 'mm_celeba_hq'
```

### Q2: 如何使用 GPU？

**A:** 确保安装了 CUDA 版本的 PyTorch，代码会自动检测 GPU。手动指定设备：

```bash
python train_causal_chain.py --device cuda
```

### Q3: 内存不足怎么办？

**A:** 减小 batch_size：

```bash
python train_causal_chain.py --batch-size 16
```

### Q4: 如何快速验证代码？

**A:** 使用调试模式，只使用少量数据：

```bash
python train_causal_chain.py --debug
```

### Q5: 数据集路径不对怎么办？

**A:** 检查 `common/config.py` 中的 `project_root` 配置，确保路径与你的实际目录一致：

```python
project_root = "D:\\code\\causality\\FrontdoorCausalChain"
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

# 5. 更新 models/__init__.py
from . import my_model
```

## 进阶用法

### 自定义数据集

继承 `DatasetLoader` 并实现自己的数据集加载器：

```python
from common.dataset_loaders import DatasetLoader

class MyDatasetLoader(DatasetLoader):
    def load_data(self, test_size=0.2, random_state=42):
        # 实现数据加载逻辑
        return train_df, valid_df
```

### 实验记录

建议使用实验跟踪工具（如 wandb、tensorboard）记录实验结果。

```python
import wandb

# 在训练开始前
wandb.init(project="my-project", config=config.__dict__)

# 在训练循环中
wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
```

### 多GPU训练

使用 PyTorch 的 `DistributedDataParallel`：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```

## 下一步

- 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解项目架构
- 查看 [models/frontdoor/](models/frontdoor/) 了解模型实现
- 尝试添加自己的模型或数据集
