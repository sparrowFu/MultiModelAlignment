"""
检查目录结构是否正确
"""
import os
import sys


def check_structure():
    """检查目录结构"""
    print("=" * 60)
    print("检查Baseline目录结构")
    print("=" * 60)

    required_files = {
        'root': [
            'train.py',
            'evaluate.py',
            'README.md',
            'QUICKSTART.md',
            'ARCHITECTURE.md',
            'REFACTORING_SUMMARY.md',
            '.gitignore',
        ],
        'common': [
            '__init__.py',
            'config.py',
            'dataset.py',
            'data.py',
            'metrics.py',
            'training.py',
        ],
        'models/clip': [
            '__init__.py',
            'config.py',
            'model.py',
            'train.py',
            'evaluate.py',
        ],
        'models/template': [
            '__init__.py',
            'config.py',
            'model.py',
            'train.py',
            'evaluate.py',
        ],
    }

    all_passed = True

    for category, files in required_files.items():
        print(f"\n检查 {category}/")
        for file in files:
            path = os.path.join(category.replace('/', os.sep), file)
            exists = os.path.exists(path)
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {file}")
            if not exists:
                all_passed = False

    print("\n" + "=" * 60)

    if all_passed:
        print("✓ 所有必需文件都存在！")
        print("\n下一步:")
        print("1. 安装依赖: pip install torch transformers timm albumentations")
        print("2. 训练模型: python train.py --model clip")
        print("3. 评估模型: python evaluate.py --model clip")
        return 0
    else:
        print("✗ 部分文件缺失，请检查！")
        return 1


if __name__ == "__main__":
    sys.exit(check_structure())
