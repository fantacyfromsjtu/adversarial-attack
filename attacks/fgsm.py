"""
FGSM (Fast Gradient Sign Method) 攻击实现
"""
import torch
import torch.nn as nn


class FGSM:
    """
    FGSM攻击方法
    
    Args:
        model: 目标模型
        epsilon: 扰动强度
        clip_min: 最小像素值
        clip_max: 最大像素值
    """
    def __init__(self, model, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(self, x, y):
        """
        生成对抗样本
        
        Args:
            x: 原始输入图像
            y: 真实标签
        
        Returns:
            x_adv: 对抗样本
            perturbation: 扰动
        """
        # 确保输入需要梯度
        x = x.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 获取梯度符号
        sign_data_grad = x.grad.sign()
        
        # 创建对抗样本
        perturbation = self.epsilon * sign_data_grad
        x_adv = x + perturbation
        
        # 裁剪到有效范围
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach(), perturbation.detach()
    
    def set_params(self, **kwargs):
        """设置攻击参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 