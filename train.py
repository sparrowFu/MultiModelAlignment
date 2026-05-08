"""
训练脚本 - 统一的训练入口
"""
import argparse
from models.clip import train_clip


def main():
    parser = argparse.ArgumentParser(description='训练baseline模型')
    parser.add_argument('--model', type=str, default='clip', choices=['clip'],
                        help='选择要训练的模型')
    parser.add_argument('--model-path', type=str, default='best.pt',
                        help='模型保存路径')

    args = parser.parse_args()

    if args.model == 'clip':
        print("训练CLIP模型...")
        train_clip(model_path=args.model_path, resume=True)
    else:
        print(f"未知模型: {args.model}")


if __name__ == "__main__":
    main()
