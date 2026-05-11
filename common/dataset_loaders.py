"""
数据集加载器 - 支持多种数据集格式
"""
import os
import pandas as pd
import pyarrow as pa
from typing import Tuple, Optional
from .config import BaseConfig


class DatasetLoader:
    """数据集加载器基类"""

    def __init__(self, config: Optional[BaseConfig] = None):
        """
        Args:
            config: 配置对象
        """
        self.config = config or BaseConfig()
        self.dataset_name = self.config.dataset_name

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载训练集和验证集

        Returns:
            train_df: 训练数据 DataFrame
            valid_df: 验证数据 DataFrame
        """
        raise NotImplementedError("子类需要实现此方法")


class Flickr30kLoader(DatasetLoader):
    """Flickr30k 数据集加载器

    数据格式: captions.txt 文件包含所有图片的描述
    格式: image_name|comment
    """

    def load_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载 Flickr30k 数据集"""
        captions_file = os.path.join(self.config.captions_path, 'captions.txt')

        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"找不到 captions.txt 文件: {captions_file}")

        # 读取 captions.txt，使用分隔符 '|'
        df = pd.read_csv(captions_file)

        # 重命名列以保持一致性
        if 'comment' in df.columns:
            df['caption'] = df['comment']
        if 'image_name' in df.columns:
            df['image'] = df['image_name']

        # 清理数据中的空格
        if 'caption' in df.columns:
            df['caption'] = df['caption'].str.strip()

        # 限制数据量（调试模式）
        max_id = df.shape[0] if not self.config.debug else 100
        df = df.iloc[:max_id]

        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        train_df = train_df.reset_index(names="original_index")
        valid_df = valid_df.reset_index(names="original_index")

        return train_df, valid_df


class MMCelebaHQLoader(DatasetLoader):
    """MM-CELEBA-HQ 数据集加载器

    数据格式:
    - 图片: images/0.jpg, images/1.jpg, ...
    - 文本: text/0.txt, text/1.txt, ... (每个文件包含10行描述)
    """

    def load_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载 MM-CELEBA-HQ 数据集"""
        image_dir = os.path.join(self.config.image_path)
        text_dir = os.path.join(self.config.captions_path)

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"找不到图片目录: {image_dir}")
        if not os.path.exists(text_dir):
            raise FileNotFoundError(f"找不到文本目录: {text_dir}")

        # 获取所有图片文件
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        # 读取对应的文本描述
        data = []
        for img_file in image_files:
            # 获取图片ID（如 0.jpg -> 0）
            img_id = os.path.splitext(img_file)[0]
            txt_file = os.path.join(text_dir, f"{img_id}.txt")

            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    captions = [line.strip() for line in f.readlines() if line.strip()]

                # 每条描述作为一个样本
                for caption in captions:
                    data.append({
                        'image': img_file,
                        'image_name': img_file,
                        'caption': caption,
                        'comment': caption
                    })

        df = pd.DataFrame(data)

        if df.empty:
            raise ValueError("没有加载到任何数据，请检查数据集路径和格式")

        # 限制数据量（调试模式）
        max_id = df.shape[0] if not self.config.debug else 100
        df = df.iloc[:max_id]

        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        train_df = train_df.reset_index(names="original_index")
        valid_df = valid_df.reset_index(names="original_index")

        return train_df, valid_df


class MSCOCO15kLoader(DatasetLoader):
    """MSCOCO-15k 数据集加载器

    数据格式: .arrow 格式，分为训练集和验证集
    - 训练集: mscoco_15k_train/
    - 验证集: mscoco_15k_test/
    """

    def load_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载 MSCOCO-15k 数据集"""
        train_path = self.config.train_path
        valid_path = self.config.valid_path

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到训练集目录: {train_path}")
        if not os.path.exists(valid_path):
            raise FileNotFoundError(f"找不到验证集目录: {valid_path}")

        # 查找 .arrow 文件
        train_arrow = self._find_arrow_file(train_path)
        valid_arrow = self._find_arrow_file(valid_path)

        if train_arrow is None:
            raise FileNotFoundError(f"训练集中找不到 .arrow 文件: {train_path}")
        if valid_arrow is None:
            raise FileNotFoundError(f"验证集中找不到 .arrow 文件: {valid_path}")

        # 读取 .arrow 文件
        train_df = self._read_arrow_file(train_arrow)
        valid_df = self._read_arrow_file(valid_arrow)

        # 标准化列名
        train_df = self._standardize_columns(train_df)
        valid_df = self._standardize_columns(valid_df)

        # 添加索引
        train_df = train_df.reset_index(names="original_index")
        valid_df = valid_df.reset_index(names="original_index")

        # 限制数据量（调试模式）
        if self.config.debug:
            train_df = train_df.iloc[:100]
            valid_df = valid_df.iloc[:50]

        return train_df, valid_df

    def _find_arrow_file(self, directory: str) -> Optional[str]:
        """在目录中查找 .arrow 文件"""
        for file in os.listdir(directory):
            if file.endswith('.arrow'):
                return os.path.join(directory, file)
        return None

    def _read_arrow_file(self, arrow_path: str) -> pd.DataFrame:
        """读取 .arrow 文件并转换为 DataFrame"""
        try:
            # 使用 pyarrow 读取文件
            table = pa.ipc.open_file(arrow_path).read_all()
            df = table.to_pandas()
            return df
        except Exception as e:
            raise IOError(f"读取 .arrow 文件失败: {arrow_path}, 错误: {e}")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名以保持一致性"""
        # 可能的列名映射
        column_mapping = {
            'image_name': 'image',
            'img_name': 'image',
            'filename': 'image',
            'text': 'caption',
            'description': 'caption',
            'comment': 'caption',
        }

        # 重命名列
        df = df.rename(columns=column_mapping)

        # 确保存在必要的列
        if 'image' not in df.columns:
            raise ValueError("数据集中缺少 'image' 列")
        if 'caption' not in df.columns:
            raise ValueError("数据集中缺少 'caption' 列")

        # 确保 image_name 列存在（用于兼容性）
        if 'image_name' not in df.columns:
            df['image_name'] = df['image']

        # 确保 comment 列存在（用于兼容性）
        if 'comment' not in df.columns:
            df['comment'] = df['caption']

        return df


def get_dataset_loader(config: Optional[BaseConfig] = None) -> DatasetLoader:
    """
    根据配置获取对应的数据集加载器

    Args:
        config: 配置对象

    Returns:
        数据集加载器实例
    """
    config = config or BaseConfig()
    dataset_name = config.dataset_name.lower()

    loader_map = {
        'flickr30k': Flickr30kLoader,
        'mm_celeba_hq': MMCelebaHQLoader,
        'mscoco_15k': MSCOCO15kLoader,
    }

    loader_class = loader_map.get(dataset_name)

    if loader_class is None:
        raise ValueError(
            f"不支持的数据集: {dataset_name}. "
            f"支持的数据集: {list(loader_map.keys())}"
        )

    return loader_class(config)


def make_train_valid_dfs(
    config: Optional[BaseConfig] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    创建训练和验证数据集（统一接口）

    Args:
        config: 配置对象
        test_size: 验证集比例
        random_state: 随机种子

    Returns:
        train_dataframe: 训练数据 DataFrame
        valid_dataframe: 验证数据 DataFrame
    """
    loader = get_dataset_loader(config)
    return loader.load_data(test_size=test_size, random_state=random_state)
