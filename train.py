"""
训练脚本 - 统一的训练入口

运行方式:
    python train.py
    python train.py --model clip
    python train.py --model frontdoor
    python train.py --model frontdoor --dataset mm_celeba_hq
"""
import argparse
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train_clip(args):
    """训练 CLIP 模型"""
    from models.clip.train import train as clip_train
    from models.clip.config import CLIPConfig

    # 创建配置并应用命令行参数
    config = CLIPConfig()
    config.dataset_name = args.dataset
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.debug:
        config.debug = True

    # 将配置应用到全局
    import common.config as config_module
    config_module.CFG = config

    print("=" * 60)
    print("训练 CLIP 模型")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"批大小: {config.batch_size}")
    print(f"训练轮数: {config.epochs}")

    # 调用 CLIP 训练函数
    clip_train(model_path=args.model_path, resume=not args.no_resume)


def train_frontdoor(args):
    """训练 FrontDoor 因果链模型"""
    from models.frontdoor.train import train as frontdoor_train
    from models.frontdoor.config import FrontDoorConfig

    # 创建配置
    config = FrontDoorConfig()

    # 应用命令行参数
    config.dataset_name = args.dataset
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.debug:
        config.debug = True

    print("=" * 60)
    print("训练 FrontDoor 因果链模型")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"批大小: {config.batch_size}")
    print(f"训练轮数: {config.epochs}")
    print(f"学习率: {config.lr}")

    # 调用 FrontDoor 训练函数
    frontdoor_train(config=config, resume=not args.no_resume)


def main():
    parser = argparse.ArgumentParser(
        description='训练多模态模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py                           # 训练 CLIP 模型（默认）
  python train.py --model frontdoor         # 训练 FrontDoor 模型
  python train.py --model frontdoor --dataset mm_celeba_hq
  python train.py --model frontdoor --batch-size 64 --epochs 10
        """
    )

    # 模型选择
    parser.add_argument(
        '--model',
        type=str,
        default='clip',
        choices=['clip', 'frontdoor'],
        help='选择要训练的模型 (clip/frontdoor)'
    )

    # 数据集选择
    parser.add_argument(
        '--dataset',
        type=str,
        default='flickr30k',
        choices=['flickr30k', 'mm_celeba_hq', 'mscoco_15k'],
        help='选择数据集 (flickr30k/mm_celeba_hq/mscoco_15k)'
    )

    # 训练参数
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批大小（覆盖配置文件）'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数（覆盖配置文件）'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学习率（覆盖配置文件）'
    )

    # 其他选项
    parser.add_argument(
        '--model-path',
        type=str,
        default='D:\\code\\causality\\FrontdoorCausalChain\\results\\clipmodel\\best_model.pt',
        help='模型保存路径'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式（使用少量数据）'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='不从 checkpoint 恢复训练'
    )

    args = parser.parse_args()

    # 根据模型类型调用相应的训练函数
    if args.model == 'clip':
        train_clip(args)
    elif args.model == 'frontdoor':
        train_frontdoor(args)
    else:
        print(f"未知模型: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
