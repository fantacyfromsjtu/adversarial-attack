"""
PGD (Projected Gradient Descent) 攻击实现
"""
import torch
import torch.nn as nn


class PGD:
    """
    PGD攻击方法
    
    Args:
        model: 目标模型
        epsilon: 扰动强度
        alpha: 步长
        num_steps: 迭代次数
        random_start: 是否随机初始化
        device: 运行设备
    """
    def __init__(self, model, epsilon=0.03, alpha=0.01, num_steps=40, random_start=True, device=None):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
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
        # 将数据移动到指定设备
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_adv = x.clone().detach()
        
        # 随机初始化
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # PGD迭代
        for _ in range(self.num_steps):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, y)
            
            self.model.zero_grad()
            loss.backward()
            
            # 更新对抗样本
            adv_grad = x_adv.grad.sign()
            x_adv = x_adv.detach() + self.alpha * adv_grad
            
            # 投影到epsilon球内
            delta = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
            x_adv = torch.clamp(x + delta, min=0.0, max=1.0).detach()
        
        perturbation = x_adv - x
        return x_adv, perturbation
    
    def set_params(self, **kwargs):
        """设置攻击参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 