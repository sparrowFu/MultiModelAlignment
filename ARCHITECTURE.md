# 架构说明文档

## 设计理念

本项目采用模块化、可扩展的设计理念，便于：
- 快速添加新的模型
- 代码复用和维护
- 统一的训练和评估接口
- 支持多种数据集格式

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    FrontdoorCausalChain                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ train.py    │  │ evaluate.py  │  │  README.md   │      │
│  │ 统一训练入口  │  │ 统一评估入口  │  │  项目文档    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                                  │
│         └─────────┬───────┘                                  │
│                   │                                          │
│  ┌────────────────▼──────────────────────────────────┐      │
│  │                    models/                         │      │
│  │              各模型实现目录                         │      │
│  ├────────────────────────────────────────────────────┤      │
│  │                                                     │      │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │      │
│  │  │    clip/    │  │  frontdoor/  │  │ template/ │ │      │
│  │  │  CLIP模型   │  │  因果链模型   │  │  新模型模板│ │      │
│  │  │             │  │              │  │           │ │      │
│  │  │ ├config.py  │  │ ├config.py   │  │ ├config.py│ │      │
│  │  │ ├model.py   │  │ ├model.py    │  │ ├model.py │ │      │
│  │  │ ├train.py   │  │ ├loss.py     │  │ ├train.py │ │      │
│  │  │ └evaluate.py│  │ ├train.py    │  │ └evaluate.py│ │     │
│  │  └─────────────┘  │ └evaluate.py │  └───────────┘ │      │
│  │                   └──────────────┘                 │      │
│  └─────────────────────────────────────────────────────┘      │
│                   │ 依赖                                     │
│                   ▼                                          │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                    common/                          │     │
│  │              共享工具和基础类                         │     │
│  ├─────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │     │
│  │  │  config.py  │  │  data.py    │  │dataset_loaders│ │    │
│  │  │  BaseConfig │  │数据处理工具  │  │  多数据集加载  │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │     │
│  │                                                      │     │
│  │  ┌─────────────┐  ┌─────────────┐                  │     │
│  │  │  dataset.py │  │  metrics.py │                  │     │
│  │  │ BaseDataset │  │ AvgMeter    │                  │     │
│  │  │ ArrowDataset│  │ get_lr      │                  │     │
│  │  └─────────────┘  └─────────────┘                  │     │
│  │                                                      │     │
│  │  ┌─────────────────────────────┐                    │     │
│  │  │        training.py          │                    │     │
│  │  │  train_epoch()              │                    │     │
│  │  │  valid_epoch()              │                    │     │
│  │  └─────────────────────────────┘                    │     │
│  │                                                      │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                        data/                        │    │
│  │              支持的数据集                            │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  flickr30k/  MM-CELEBA-HQ/  mscoco_15k/              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 模块依赖关系

```
train.py / train_causal_chain.py
    │
    ├──> models.frontdoor
    │        │
    │        ├──> common (共享工具)
    │        └──> models.clip (编码器)
    │             └──> common (共享工具)
    │
    └──> models.clip
             │
             ├──> common (共享工具)
             └──> transformers, timm (外部库)
```

## 数据流

### 数据集加载流程

```
1. 选择数据集 (dataset_name)
   ↓
2. DatasetLoader 自动选择对应的加载器
   ├─> Flickr30kLoader (captions.txt)
   ├─> MMCelebaHQLoader (图片-txt 对应)
   └─> MSCOCO15kLoader (.arrow 格式)
   ↓
3. 构建 DataFrame
   ├─> flickr30k: 读取 captions.txt
   ├─> mm_celeba_hq: 扫描图片和txt文件
   └─> mscoco_15k: 读取.arrow文件
   ↓
4. 划分训练集/验证集
   ↓
5. 构建 DataLoader
   ├─> BaseDataset (常规图片)
   └─> ArrowDataset (.arrow格式)
   ↓
6. 返回训练/验证 DataLoader
```

### FrontDoor 训练流程

```
1. 读取数据 (make_train_valid_dfs)
   ├─> 自动选择数据集加载器
   └─> 返回 train_df, valid_df
   ↓
2. 创建数据加载器 (build_loaders)
   ├─> 自动选择 Dataset 类型
   └─> 返回 train_loader, valid_loader
   ↓
3. 加载预训练编码器 (ImageEncoder, TextEncoder)
   ↓
4. 初始化因果模型 (FrontDoorCausalModel)
   ↓
5. 训练循环 (train_epoch)
   │
   ├──> 前向传播 (model.forward)
   │    ├─> 图像编码
   │    ├─> 文本编码
   │    ├─> Shared/Private 分解
   │    ├─> 共享语义计算
   │    └─> 因果效应估计
   │
   ├──> 计算损失 (FrontDoorLoss)
   │    ├─> 对齐损失 (alignment_loss)
   │    ├─> 正交损失 (orthogonal_loss)
   │    ├─> 对比损失 (contrastive_loss)
   │    └─> 重建损失 (reconstruction_loss)
   │
   ├──> 反向传播
   └──> 更新参数
   ↓
6. 验证 (valid_epoch)
   ↓
7. 保存最佳模型
```

## 类继承关系

```
BaseConfig (common/config.py)
    │
    ├──> CLIPConfig (models/clip/config.py)
    ├──> FrontDoorConfig (models/frontdoor/config.py)
    └──> TemplateConfig (models/template/config.py)

BaseDataset (common/dataset.py)
    │
    └──> 用于所有常规图片数据集

ArrowDataset (common/dataset.py)
    │
    └──> 用于 .arrow 格式数据集

DatasetLoader (common/dataset_loaders.py)
    │
    ├──> Flickr30kLoader
    ├──> MMCelebaHQLoader
    └──> MSCOCO15kLoader
```

## 数据集支持

### Flickr30k

```
数据集结构:
flickr30k/
├── flickr30k_images/
│   ├── 1000092795.jpg
│   ├── 10002456.jpg
│   └── ...
└── captions.txt

captions.txt 格式:
image_name|comment
1000092795.jpg|Two men are playing basketball.
1000092795.jpg|A game of basketball.
...
```

### MM-CELEBA-HQ

```
数据集结构:
MM-CELEBA-HQ/
├── images/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── text/
    ├── 0.txt (每行一条描述，共10条)
    ├── 1.txt
    └── ...
```

### MSCOCO-15k

```
数据集结构:
mscoco_15k/
├── mscoco_15k_train/
│   └── data.arrow
└── mscoco_15k_test/
    └── data.arrow

.arrow 文件包含:
- image: 图像数据 (bytes)
- caption: 文本描述 (str)
```

## 扩展指南

### 添加新数据集

1. 在 `common/dataset_loaders.py` 中创建新的加载器类：

```python
class YourDatasetLoader(DatasetLoader):
    def load_data(self, test_size=0.2, random_state=42):
        # 实现数据加载逻辑
        return train_df, valid_df
```

2. 在 `get_dataset_loader` 函数中注册：

```python
loader_map = {
    'flickr30k': Flickr30kLoader,
    'mm_celeba_hq': MMCelebaHQLoader,
    'mscoco_15k': MSCOCO15kLoader,
    'your_dataset': YourDatasetLoader,  # 添加这里
}
```

3. 在 `BaseConfig` 中添加路径配置（如果需要特殊路径）

### 添加新模型

参考 `models/template/` 目录，实现以下文件：

- `config.py` - 模型配置（继承 `BaseConfig`）
- `model.py` - 模型定义
- `train.py` - 训练脚本
- `evaluate.py` - 评估脚本

## 最佳实践

### 1. 配置管理

- 使用配置类集中管理参数
- 支持命令行参数覆盖
- 记录实验配置

### 2. 代码复用

- 优先使用 `common` 中的工具
- 避免重复造轮子
- 保持接口一致

### 3. 模块化设计

- 单一职责原则
- 清晰的模块边界
- 易于测试和维护

### 4. 数据集支持

- 统一的数据加载接口
- 自动选择数据集类型
- 灵活的格式支持

## 性能考虑

### 内存优化

- 使用 DataLoader 的多进程加载
- 适当的 batch_size
- 及时清理无用变量

### 计算优化

- 混合精度训练
- 梯度累积
- 学习率调度

### I/O 优化

- 预处理数据
- 缓存编码结果
- 异步数据加载
