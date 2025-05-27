"""
C&W (Carlini & Wagner) 攻击简化实现
"""
import torch
import torch.nn as nn


class CW:
    """
    C&W L2攻击方法（简化版）
    """
    def __init__(self, model, c=1e-4, max_iterations=100):
        self.model = model
        self.c = c
        self.max_iterations = max_iterations
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(self, x, y):
        """
        生成对抗样本（简化版，使用类似FGSM的方法）
        """
        x = x.clone().detach().requires_grad_(True)
        
        # 简化的CW攻击：多次迭代的梯度攻击
        best_adv = x.clone()
        best_loss = float('inf')
        
        for i in range(self.max_iterations):
            x.requires_grad = True
            outputs = self.model(x)
            
            # 计算损失：最大化错误分类的概率
            loss = -self.criterion(outputs, y)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 更新
            with torch.no_grad():
                grad = x.grad.sign()
                x = x + 0.01 * grad
                x = torch.clamp(x, 0, 1)
                
                # 检查是否找到更好的对抗样本
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_adv = x.clone()
                    
                # 如果已经成功攻击，提前退出
                pred = outputs.argmax(1)
                if pred != y:
                    break
        
        perturbation = best_adv - x
        return best_adv.detach(), perturbation.detach()
    
    def set_params(self, **kwargs):
        """设置攻击参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 