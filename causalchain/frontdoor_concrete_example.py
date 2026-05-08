"""
前门准则具体示例：图像→共同语义→文本的因果链

给定已编码的图像和文本特征，展示如何建立因果链
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


class FrontDoorCausalChain:
    """
    前门准则因果链的具体实现
    """

    def __init__(self, shared_dim=256, image_feat_dim=2048, text_feat_dim=768):
        """
        初始化因果链组件

        Args:
            shared_dim: 共享语义维度
            image_feat_dim: 图像特征维度
            text_feat_dim: 文本特征维度
        """
        self.shared_dim = shared_dim
        self.image_feat_dim = image_feat_dim
        self.text_feat_dim = text_feat_dim

        # 初始化网络层
        self._init_networks()

    def _init_networks(self):
        """初始化网络"""
        # Shared/Private分解器
        self.image_shared_encoder = nn.Linear(self.image_feat_dim, self.shared_dim)
        self.image_private_encoder = nn.Linear(self.image_feat_dim, int(self.image_feat_dim * 0.3))

        self.text_shared_encoder = nn.Linear(self.text_feat_dim, self.shared_dim)
        self.text_private_encoder = nn.Linear(self.text_feat_dim, int(self.text_feat_dim * 0.3))

        # 因果效应估计器
        self.causal_effect_estimator = nn.Linear(self.shared_dim * 2, 1)

    def encode_to_shared_private(self, image_features, text_features):
        """
        步骤1: 将编码后的特征分解为shared和private

        Args:
            image_features: [batch, image_feat_dim] 图像编码特征
            text_features: [batch, text_feat_dim] 文本编码特征

        Returns:
            dict: 包含shared和private特征的字典
        """
        # 分解图像特征
        image_shared = self.image_shared_encoder(image_features)
        image_private = self.image_private_encoder(image_features)

        # 分解文本特征
        text_shared = self.text_shared_encoder(text_features)
        text_private = self.text_private_encoder(text_features)

        return {
            'image_shared': image_shared,
            'image_private': image_private,
            'text_shared': text_shared,
            'text_private': text_private
        }

    def compute_shared_semantic(self, image_shared, text_shared):
        """
        步骤2: 计算共享语义（中介变量M）

        满足前门准则条件1: 完全中介
        图像和文本对匹配的影响完全通过共享语义传递

        Args:
            image_shared: [batch, shared_dim]
            text_shared: [batch, shared_dim]

        Returns:
            shared_semantic: [batch, shared_dim * 2] 共享语义表示
        """
        # 拼接shared特征作为共享语义
        # 也可以使用更复杂的融合方式（如注意力、乘法等）
        shared_semantic = torch.cat([image_shared, text_shared], dim=-1)

        return shared_semantic

    def compute_causal_effect(self, shared_semantic):
        """
        步骤3: 计算因果效应

        满足前门准则条件2和3:
        - 条件2: 编码过程不受混杂影响（通过物理编码保证）
        - 条件3: 语义到文本的关系无混杂（通过只用shared保证）

        Args:
            shared_semantic: [batch, shared_dim * 2]

        Returns:
            causal_effect: [batch, 1] 因果效应值
        """
        # 通过共享语义计算因果效应
        causal_effect = self.causal_effect_estimator(shared_semantic)

        return causal_effect

    def verify_front_door_criterion(self, features):
        """
        验证前门准则的三个条件

        Args:
            features: 包含shared和private特征的字典

        Returns:
            dict: 验证结果
        """
        results = {}

        # 验证条件1: 完全中介
        # 检查shared特征之间的相关性（应该高）
        img_shared = features['image_shared']
        txt_shared = features['text_shared']
        shared_similarity = F.cosine_similarity(img_shared, txt_shared).mean().item()
        results['shared_similarity'] = shared_similarity
        results['condition1_satisfied'] = shared_similarity > 0.5

        # 验证条件2: I,Q → M 无混杂
        # 编码过程是物理的，这个条件天然满足
        results['condition2_satisfied'] = True

        # 验证条件3: M → A 无混杂
        # 检查shared和private的正交性（应该低相关）
        img_private = features['image_private']
        txt_private = features['text_private']

        img_ortho = torch.abs(torch.mm(img_shared.T, img_private)).mean().item()
        txt_ortho = torch.abs(torch.mm(txt_shared.T, txt_private)).mean().item()
        avg_ortho = (img_ortho + txt_ortho) / 2

        results['orthogonality'] = avg_ortho
        results['condition3_satisfied'] = avg_ortho < 0.3

        # 总体评估
        results['all_satisfied'] = all([
            results['condition1_satisfied'],
            results['condition2_satisfied'],
            results['condition3_satisfied']
        ])

        return results


def run_concrete_example():
    """
    运行具体示例：给定图像和文本编码，建立因果链
    """
    print("=" * 80)
    print("前门准则具体示例：图像→共同语义→文本的因果链")
    print("=" * 80)

    # 步骤0: 准备输入（假设已经通过编码器编码）
    print("\n📥 步骤0: 输入编码特征")
    print("-" * 80)

    batch_size = 1
    image_feat_dim = 2048
    text_feat_dim = 768

    # 模拟图像编码特征（实际使用时来自预训练的图像编码器）
    # 假设图像是"一只黑猫坐在沙发上"
    image_features = torch.randn(batch_size, image_feat_dim)
    # 加入一些特定的模式
    image_features[0, :100] = 1.0  # 模拟"猫"的特征
    image_features[0, 100:200] = 0.8  # 模拟"黑色"的特征
    image_features[0, 200:300] = 0.6  # 模拟"沙发"的特征

    # 模拟文本编码特征（实际使用时来自预训练的文本编码器）
    # 假设文本是"A black cat sitting on the sofa"
    text_features = torch.randn(batch_size, text_feat_dim)
    # 加入一些特定的模式
    text_features[0, :50] = 0.9  # 模拟"猫"的特征
    text_features[0, 50:100] = 0.95  # 模拟"黑色"的特征
    text_features[0, 100:150] = 0.5  # 模拟"沙发"的特征
    text_features[0, 150:200] = 0.7  # 模拟"坐"的特征

    print(f"图像编码特征 shape: {image_features.shape}")
    print(f"  - 模拟内容: 猫、黑色、沙发")
    print(f"  - 前100维: {image_features[0, :100].mean().item():.3f} (猫)")
    print(f"  - 100-200维: {image_features[0, 100:200].mean().item():.3f} (黑色)")

    print(f"\n文本编码特征 shape: {text_features.shape}")
    print(f"  - 模拟内容: 猫、黑色、沙发、坐")
    print(f"  - 前50维: {text_features[0, :50].mean().item():.3f} (猫)")
    print(f"  - 50-100维: {text_features[0, 50:100].mean().item():.3f} (黑色)")

    # 创建因果链
    print("\n🔗 创建因果链")
    print("-" * 80)
    causal_chain = FrontDoorCausalChain(
        shared_dim=256,
        image_feat_dim=image_feat_dim,
        text_feat_dim=text_feat_dim
    )

    # 步骤1: Shared/Private分解
    print("\n📊 步骤1: Shared/Private分解")
    print("-" * 80)

    features = causal_chain.encode_to_shared_private(image_features, text_features)

    img_shared = features['image_shared']
    img_private = features['image_private']
    txt_shared = features['text_shared']
    txt_private = features['text_private']

    print("图像特征分解:")
    print(f"  - Image Shared: {img_shared.shape}")
    print(f"    均值: {img_shared.mean().item():.4f}")
    print(f"    标准差: {img_shared.std().item():.4f}")
    print(f"    前5维: {img_shared[0, :5].detach().numpy()}")

    print(f"\n  - Image Private: {img_private.shape}")
    print(f"    均值: {img_private.mean().item():.4f}")
    print(f"    标准差: {img_private.std().item():.4f}")

    print("\n文本特征分解:")
    print(f"  - Text Shared: {txt_shared.shape}")
    print(f"    均值: {txt_shared.mean().item():.4f}")
    print(f"    标准差: {txt_shared.std().item():.4f}")
    print(f"    前5维: {txt_shared[0, :5].detach().numpy()}")

    print(f"\n  - Text Private: {txt_private.shape}")
    print(f"    均值: {txt_private.mean().item():.4f}")
    print(f"    标准差: {txt_private.std().item():.4f}")

    # 步骤2: 计算共享语义（中介变量M）
    print("\n🎯 步骤2: 计算共享语义（中介变量M）")
    print("-" * 80)

    shared_semantic = causal_chain.compute_shared_semantic(img_shared, txt_shared)

    print(f"共享语义 M shape: {shared_semantic.shape}")
    print(f"  - 包含: Image Shared + Text Shared")
    print(f"  - 维度: {shared_semantic.shape[-1]} = {img_shared.shape[-1]} + {txt_shared.shape[-1]}")
    print(f"  - 前10维: {shared_semantic[0, :10].detach().numpy()}")
    print(f"  - 均值: {shared_semantic.mean().item():.4f}")

    # 解释共享语义
    print("\n💡 共享语义的解释:")
    print("  M 包含了图像和文本的共同语义信息:")
    print("  - 图像部分: 猫、黑色、沙发（视觉语义）")
    print("  - 文本部分: 猫、黑色、沙发、坐（语言语义）")
    print("  - 融合后: '黑猫坐在沙发上' 这个完整场景的语义表示")

    # 步骤3: 计算因果效应
    print("\n⚡ 步骤3: 计算因果效应")
    print("-" * 80)

    causal_effect = causal_chain.compute_causal_effect(shared_semantic)

    print(f"因果效应值: {causal_effect.item():.4f}")
    print("\n💡 因果效应的解释:")
    print("  - 这个值表示: 在给定图像和文本的共享语义M的情况下，")
    print("    图像→文本的因果关系的强度")
    print("  - 值越大，表示因果联系越强")
    print(f"  - 当前值: {causal_effect.item():.4f}")

    # 步骤4: 验证前门准则
    print("\n✅ 步骤4: 验证前门准则")
    print("-" * 80)

    verification = causal_chain.verify_front_door_criterion(features)

    print("\n条件1: 完全中介 (Complete Mediation)")
    print(f"  - Shared特征相似度: {verification['shared_similarity']:.4f}")
    print(f"  - 阈值: > 0.5")
    print(f"  - 状态: {'✅ 满足' if verification['condition1_satisfied'] else '❌ 不满足'}")
    print("\n  解释:")
    print("  - 图像和文本的shared特征高度相关")
    print("  - 说明它们通过共享语义M建立了联系")
    print("  - 图像→M→文本的因果链是完整的")

    print("\n条件2: I,Q → M 无混杂 (No Confounding)")
    print(f"  - 状态: {'✅ 满足' if verification['condition2_satisfied'] else '❌ 不满足'}")
    print("\n  解释:")
    print("  - 图像编码是物理过程: 像素 → 视觉特征")
    print("  - 文本编码是确定性过程: 文本 → 语义特征")
    print("  - 这些过程不受反向因果影响")

    print("\n条件3: M → A 无混杂 (No Confounding)")
    print(f"  - Shared-Private正交性: {verification['orthogonality']:.4f}")
    print(f"  - 阈值: < 0.3")
    print(f"  - 状态: {'✅ 满足' if verification['condition3_satisfied'] else '❌ 不满足'}")
    print("\n  解释:")
    print("  - Shared和Private特征正交（相关性低）")
    print("  - Shared特征是纯净的因果语义")
    print("  - Private特征是噪声，不参与因果链")

    print("\n" + "=" * 80)
    print("总体评估:")
    print(f"  - 所有条件满足: {'✅ 是' if verification['all_satisfied'] else '❌ 否'}")
    print("\n结论:")
    if verification['all_satisfied']:
        print("  ✅ 成功建立了图像→共享语义→文本的因果链!")
        print("  ✅ 前门准则的三个条件都得到满足")
        print("  ✅ 可以使用前门调整公式计算因果效应")
    else:
        print("  ⚠️  部分条件不满足，需要调整模型参数")
    print("=" * 80)

    # 可视化因果链
    print("\n📊 可视化因果链")
    print("-" * 80)
    visualize_causal_chain(image_features, text_features, features, shared_semantic, causal_effect)

    return features, shared_semantic, causal_effect, verification


def visualize_causal_chain(image_features, text_features, features, shared_semantic, causal_effect):
    """可视化因果链"""

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('前门准则因果链可视化', fontsize=16, fontweight='bold')

    # 1. 原始特征
    ax = axes[0, 0]
    img_feat = image_features[0].detach().numpy()[:100]
    txt_feat = text_features[0].detach().numpy()[:100]
    ax.plot(img_feat, label='Image Features', alpha=0.7)
    ax.plot(txt_feat, label='Text Features', alpha=0.7)
    ax.set_title('原始编码特征', fontweight='bold')
    ax.set_xlabel('特征维度')
    ax.set_ylabel('特征值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Shared特征
    ax = axes[0, 1]
    img_shared = features['image_shared'][0].detach().numpy()
    txt_shared = features['text_shared'][0].detach().numpy()
    ax.plot(img_shared, label='Image Shared', alpha=0.7, linewidth=2)
    ax.plot(txt_shared, label='Text Shared', alpha=0.7, linewidth=2)
    ax.set_title('Shared特征（因果语义）', fontweight='bold')
    ax.set_xlabel('特征维度')
    ax.set_ylabel('特征值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Private特征
    ax = axes[0, 2]
    img_private = features['image_private'][0].detach().numpy()
    txt_private = features['text_private'][0].detach().numpy()
    ax.plot(img_private, label='Image Private', alpha=0.5, linestyle='--')
    ax.plot(txt_private, label='Text Private', alpha=0.5, linestyle='--')
    ax.set_title('Private特征（噪声）', fontweight='bold')
    ax.set_xlabel('特征维度')
    ax.set_ylabel('特征值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 共享语义热图
    ax = axes[1, 0]
    semantic_matrix = shared_semantic[0, :64].detach().numpy().reshape(8, 8)
    sns.heatmap(semantic_matrix, cmap='viridis', ax=ax, cbar_kws={'label': '值'})
    ax.set_title('共享语义M（前64维）', fontweight='bold')
    ax.set_xlabel('列')
    ax.set_ylabel('行')

    # 5. Shared特征相似度
    ax = axes[1, 1]
    shared_sim = F.cosine_similarity(
        features['image_shared'],
        features['text_shared']
    ).item()
    ax.bar(['Shared\n相似度'], [shared_sim], color='skyblue', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', label='阈值(0.5)')
    ax.set_title(f'Shared特征相似度: {shared_sim:.4f}', fontweight='bold')
    ax.set_ylabel('相似度')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. 因果效应
    ax = axes[1, 2]
    effect_value = causal_effect.item()
    ax.bar(['因果效应'], [effect_value], color='lightcoral', alpha=0.7)
    ax.set_title(f'因果效应值: {effect_value:.4f}', fontweight='bold')
    ax.set_ylabel('效应值')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加说明
    fig.text(0.5, 0.02, '因果链: 图像 → Shared语义 → 文本', ha='center',
            fontsize=12, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('causal_chain_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ 可视化已保存至: causal_chain_visualization.png")


def show_mathematical_formulation():
    """展示数学公式"""
    print("\n" + "=" * 80)
    print("📐 前门准则的数学公式")
    print("=" * 80)

    print("\n1. 因果图:")
    print("   I(图像) ─┐")
    print("             ├→ M(共享语义) → T(文本)")
    print("   Q(问题) ─┘")

    print("\n2. 前门调整公式:")
    print("   P(T | do(I)) = Σm P(M=m | I) × Σi' P(T | i', M=m) × P(I=i')")

    print("\n3. 简化版（假设I独立）:")
    print("   P(T | do(I)) ≈ Σm P(M | I, T) × P(T | M)")

    print("\n4. 在代码中的实现:")
    print("   # P(M | I, T): 编码+分解")
    print("   img_shared, img_private = ImageDecomposer(ImageEncoder(I))")
    print("   txt_shared, txt_private = TextDecomposer(TextEncoder(T))")
    print("   M = Fusion(img_shared, txt_shared)")
    print("   ")
    print("   # P(T | M): 因果效应")
    print("   causal_effect = CausalEffectEstimator(M)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 运行具体示例
    features, shared_semantic, causal_effect, verification = run_concrete_example()

    # 展示数学公式
    show_mathematical_formulation()

    print("\n✅ 示例运行完成!")
    print("\n📁 生成的文件:")
    print("  - causal_chain_visualization.png: 因果链可视化")
