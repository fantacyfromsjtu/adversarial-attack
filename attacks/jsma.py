"""
JSMA (Jacobian-based Saliency Map Attack) 攻击简化实现
"""
import torch
import torch.nn as nn


class JSMA:
    """
    JSMA攻击方法（简化版）
    """
    def __init__(self, model, theta=0.1, max_iterations=100):
        self.model = model
        self.theta = theta
        self.max_iterations = max_iterations
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(self, x, y):
        """
        生成对抗样本（简化版）
        """
        x_adv = x.clone().detach()
        
        for i in range(self.max_iterations):
            x_adv.requires_grad = True
            
            # 前向传播
            outputs = self.model(x_adv)
            
            # 检查是否已经成功攻击
            pred = outputs.argmax(1)
            if pred != y:
                break
            
            # 计算损失
            loss = -self.criterion(outputs, y)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 找到影响最大的像素并修改
            with torch.no_grad():
                grad = x_adv.grad.abs()
                flat_grad = grad.view(-1)
                _, idx = flat_grad.topk(1)
                
                # 修改该像素
                flat_x_adv = x_adv.view(-1)
                if flat_x_adv[idx] < 0.5:
                    flat_x_adv[idx] = torch.clamp(flat_x_adv[idx] + self.theta, 0, 1)
                else:
                    flat_x_adv[idx] = torch.clamp(flat_x_adv[idx] - self.theta, 0, 1)
                
                x_adv = flat_x_adv.view_as(x)
        
        perturbation = x_adv - x
        return x_adv.detach(), perturbation.detach()
    
    def set_params(self, **kwargs):
        """设置攻击参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 