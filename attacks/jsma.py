"""
JSMA (Jacobian-based Saliency Map Attack) 攻击简化实现
"""
import torch
import torch.nn as nn
import numpy as np


class JSMA:
    """
    JSMA攻击方法（简化版）
    
    Args:
        model: 目标模型
        theta: 扰动强度
        gamma: 最大扰动比例
        max_iterations: 最大迭代次数
    """
    def __init__(self, model, theta=1.0, gamma=0.1, max_iterations=2000):
        self.model = model
        self.theta = theta
        self.gamma = gamma
        self.max_iterations = max_iterations
    
    def generate(self, x, y):
        """
        生成对抗样本
        
        Args:
            x: 原始输入图像 (batch_size=1)
            y: 真实标签
        
        Returns:
            x_adv: 对抗样本
            perturbation: 扰动
        """
        # 简化实现：使用FGSM的变体
        x_adv = x.clone().detach()
        n_pixels = x.numel()
        max_pixels = int(n_pixels * self.gamma)
        
        for i in range(min(self.max_iterations, max_pixels)):
            x_adv.requires_grad = True
            
            # 前向传播
            outputs = self.model(x_adv)
            
            # 计算目标类别的梯度
            target_class = outputs.argmax(1)
            if target_class != y:
                # 已经成功攻击
                break
            
            # 计算损失
            loss = -outputs[0, y]  # 最小化正确类别的分数
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 找到影响最大的像素
            grad = x_adv.grad.abs()
            flat_grad = grad.view(-1)
            _, idx = flat_grad.topk(1)
            
            # 修改该像素
            with torch.no_grad():
                flat_x_adv = x_adv.view(-1)
                if flat_x_adv[idx] < 0.5:
                    flat_x_adv[idx] = min(1.0, flat_x_adv[idx] + self.theta)
                else:
                    flat_x_adv[idx] = max(0.0, flat_x_adv[idx] - self.theta)
                
                x_adv = flat_x_adv.view_as(x)
        
        perturbation = x_adv - x
        return x_adv.detach(), perturbation.detach()
    
    def set_params(self, **kwargs):
        """设置攻击参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 