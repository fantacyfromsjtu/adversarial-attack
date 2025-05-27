"""
可视化工具函数
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from .data_loader import denormalize_cifar10
from config import FIGURE_SIZE, SUBPLOT_ADJUST

# 确保matplotlib配置生效
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def visualize_attack_results(original, adversarial, perturbation, 
                           original_pred, adversarial_pred, classes):
    """
    可视化攻击结果
    
    Args:
        original: 原始图像
        adversarial: 对抗样本
        perturbation: 扰动
        original_pred: 原始预测
        adversarial_pred: 对抗预测
        classes: 类别名称列表
    
    Returns:
        fig: matplotlib图形对象
    """
    # 确保图像在CPU上并反归一化
    original = denormalize_cifar10(original.cpu()).squeeze()
    adversarial = denormalize_cifar10(adversarial.cpu()).squeeze()
    perturbation = perturbation.cpu().squeeze()
    
    # 转换为numpy数组
    original = original.permute(1, 2, 0).numpy()
    adversarial = adversarial.permute(1, 2, 0).numpy()
    
    # 计算扰动的显示范围
    if len(perturbation.shape) == 3:
        perturbation = perturbation.permute(1, 2, 0).numpy()
    else:
        perturbation = perturbation.numpy()
    
    # 创建图形，使用较小的尺寸
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    # 原始图像
    axes[0].imshow(original)
    axes[0].set_title(f'原始图像\n预测: {classes[original_pred]}', fontsize=12)
    axes[0].axis('off')
    
    # 扰动（放大显示）
    if perturbation.max() > perturbation.min():
        perturbation_display = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
    else:
        perturbation_display = perturbation
    axes[1].imshow(perturbation_display)
    axes[1].set_title('扰动\n(放大显示)', fontsize=12)
    axes[1].axis('off')
    
    # 对抗样本
    axes[2].imshow(adversarial)
    axes[2].set_title(f'对抗样本\n预测: {classes[adversarial_pred]}', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    绘制训练历史
    
    Args:
        history: 包含训练历史的字典，应包含'train_loss', 'train_acc', 'val_loss', 'val_acc'
    
    Returns:
        fig: matplotlib图形对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # 绘制损失
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('训练轮数', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.set_title('训练和验证损失曲线', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('训练轮数', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('训练和验证准确率曲线', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_attack_success_rate(attack_methods, success_rates):
    """
    绘制不同攻击方法的成功率
    
    Args:
        attack_methods: 攻击方法名称列表
        success_rates: 对应的成功率列表
    
    Returns:
        fig: matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 使用更好看的颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax.bar(attack_methods, success_rates, color=colors[:len(attack_methods)])
    
    # 在柱状图上添加数值
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('攻击成功率 (%)', fontsize=12)
    ax.set_title('不同攻击方法的成功率比较', fontsize=14)
    ax.set_ylim(0, max(success_rates) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    # 调整x轴标签角度
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig 