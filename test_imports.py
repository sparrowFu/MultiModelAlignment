"""
测试所有模块导入是否正常
"""
import sys


def test_imports():
    """测试所有模块的导入"""
    print("测试模块导入...\n")

    tests = []

    # 测试common模块
    print("1. 测试common模块...")
    try:
        from common import BaseConfig, AvgMeter, get_lr
        from common import BaseDataset, get_transforms
        from common import make_train_valid_dfs, build_loaders
        from common import train_epoch, valid_epoch
        print("   ✅ common模块导入成功")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ common模块导入失败: {e}")
        tests.append(False)

    # 测试CLIP模型
    print("\n2. 测试CLIP模型...")
    try:
        from models.clip import CLIPConfig
        from models.clip.model import CLIPModel, ImageEncoder, TextEncoder, ProjectionHead
        from models.clip import train_clip, evaluate_clip
        print("   ✅ CLIP模型导入成功")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ CLIP模型导入失败: {e}")
        tests.append(False)

    # 测试模板
    print("\n3. 测试模板...")
    try:
        from models.template import NewModelConfig
        from models.template.model import NewModel
        print("   ✅ 模板导入成功")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ 模板导入失败: {e}")
        tests.append(False)

    # 测试配置
    print("\n4. 测试配置...")
    try:
        from common import BaseConfig
        config = BaseConfig()
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'device')
        print(f"   ✅ 配置测试成功")
        print(f"      - batch_size: {config.batch_size}")
        print(f"      - device: {config.device}")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ 配置测试失败: {e}")
        tests.append(False)

    # 测试模型创建
    print("\n5. 测试模型创建...")
    try:
        from models.clip.model import CLIPModel
        import torch
        model = CLIPModel()
        assert isinstance(model, torch.nn.Module)
        print("   ✅ CLIP模型创建成功")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ CLIP模型创建失败: {e}")
        tests.append(False)

    # 总结
    print("\n" + "="*50)
    passed = sum(tests)
    total = len(tests)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print(f"⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())
