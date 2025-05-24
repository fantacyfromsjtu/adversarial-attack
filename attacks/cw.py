"""
C&W (Carlini & Wagner) 攻击简化实现
"""
import torch
import torch.nn as nn
import torch.optim as optim


class CW:
    """
    C&W L2攻击方法（简化版）
    
    Args:
        model: 目标模型
        c: 常数参数
        kappa: 置信度参数
        max_iterations: 最大迭代次数
        learning_rate: 学习率
    """
    def __init__(self, model, c=1e-4, kappa=0, max_iterations=1000, learning_rate=0.01):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
    
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
        # 将输入转换到tanh空间
        x_tanh = self._to_tanh_space(x)
        w = x_tanh.clone().detach().requires_grad_(True)
        
        # 优化器
        optimizer = optim.Adam([w], lr=self.learning_rate)
        
        best_adv = x.clone()
        best_dist = float('inf')
        
        for step in range(self.max_iterations):
            # 将w转换回图像空间
            x_adv = self._from_tanh_space(w)
            
            # 计算模型输出
            outputs = self.model(x_adv)
            
            # 计算CW损失
            l2_dist = torch.sum((x_adv - x) ** 2)
            
            # f函数：最大化正确类别和其他类别的差距
            real = outputs.gather(1, y.view(-1, 1)).squeeze(1)
            other = torch.max((1 - torch.eye(outputs.size(1))[y]).to(x.device) * outputs, dim=1)[0]
            loss_f = torch.clamp(real - other + self.kappa, min=0.0)
            
            # 总损失
            loss = l2_dist + self.c * loss_f.sum()
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 保存最佳结果
            if l2_dist < best_dist and (outputs.argmax(1) != y).any():
                best_dist = l2_dist
                best_adv = x_adv.clone()
        
        perturbation = best_adv - x
        return best_adv.detach(), perturbation.detach()
    
    def _to_tanh_space(self, x):
        """将图像从[0,1]转换到tanh空间"""
        return torch.atanh(2 * x - 1)
    
    def _from_tanh_space(self, w):
        """将图像从tanh空间转换回[0,1]"""
        return (torch.tanh(w) + 1) / 2
    
    def set_params(self, **kwargs):
        """设置攻击参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 