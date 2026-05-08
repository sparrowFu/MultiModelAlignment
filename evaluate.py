"""
评估脚本 - 统一的评估入口
"""
import argparse
from models.clip import evaluate_clip


def main():
    parser = argparse.ArgumentParser(description='评估baseline模型')
    parser.add_argument('--model', type=str, default='clip', choices=['clip'],
                        help='选择要评估的模型')
    parser.add_argument('--model-path', type=str, default='best.pt',
                        help='模型权重路径')
    parser.add_argument('--query', type=str, default='a group of people dancing in a party',
                        help='查询文本')

    args = parser.parse_args()

    if args.model == 'clip':
        print("评估CLIP模型...")
        evaluate_clip.evaluate(model_path=args.model_path, query=args.query)
    else:
        print(f"未知模型: {args.model}")


if __name__ == "__main__":
    main()
