"""
JSMA (Jacobian-based Saliency Map Attack) 攻击简化实现
"""
import torch
import torch.nn as nn


class JSMA:
    """
    JSMA攻击方法（无目标攻击）
    
    参数:
        model: 目标模型
        theta: 每次修改的扰动大小
        gamma: 修改像素的比例上限 (0-1)
        max_iter: 最大迭代次数
    """
    def __init__(self, model, theta=0.1, gamma=1.0, max_iter=100, device=None):
        self.model = model
        self.theta = theta
        self.gamma = gamma
        self.max_iter = max_iter
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def generate(self, x, y):
        """
        生成对抗样本
        
        参数:
            x: 原始输入图像 [B, C, H, W]
            y: 真实标签 [B]
            
        返回:
            x_adv: 对抗样本
            perturbation: 扰动
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        # 确保处理单个样本
        if x.dim() == 4 and x.size(0) > 1:
            raise ValueError("JSMA当前仅支持单个样本处理")
            
        # 获取原始图像的形状
        x_adv = x.clone().detach()
        batch_size, channels, height, width = x.shape
        
        # 计算最大修改像素数
        max_pixels = int(self.gamma * channels * height * width)
        
        # 创建修改掩码，记录已修改的像素
        modified = torch.zeros_like(x, dtype=torch.bool, device=self.device)
        modified_pixels = 0
        
        # 迭代修改像素
        for _ in range(self.max_iter):
            # 检查当前预测
            with torch.no_grad():
                outputs = self.model(x_adv)
                current_pred = outputs.argmax(dim=1).item()
            
            # 如果攻击成功，则停止
            if current_pred != y.item():
                break
                
            # 计算梯度 - 对当前真实类别的梯度
            # 对于每次迭代使用新的计算图
            x_temp = x_adv.clone().detach().requires_grad_(True)
            
            # 前向传播
            outputs = self.model(x_temp)
            true_class_score = outputs[0, y.item()]
            
            # 计算对真实类别输出的梯度
            self.model.zero_grad()
            true_class_score.backward()
            
            # 获取梯度
            gradients = x_temp.grad.clone()
            
            # 创建显著图 - 简化版本
            # 我们希望找到减少真实类别概率的像素
            # 对于无目标攻击，我们寻找正梯度的像素，然后减少它们的值
            saliency_map = gradients.clone()
            
            # 只保留正梯度（增加这些像素会增加真实类别概率，所以我们要减少它们）
            saliency_map = torch.where(saliency_map > 0, saliency_map, torch.zeros_like(saliency_map))
            
            # 将已修改的像素排除
            saliency_map = torch.where(modified, torch.tensor(-float('inf'), device=self.device), saliency_map)
            
            # 如果没有合适的像素可以修改，则停止攻击
            if (saliency_map > 0).sum() == 0:
                break
                
            # 找到具有最大显著值的像素
            flat_saliency = saliency_map.view(-1)
            max_idx = torch.argmax(flat_saliency)
            
            # 转换为原始维度索引
            idx_c = (max_idx // (height * width)) % channels
            idx_h = (max_idx // width) % height
            idx_w = max_idx % width
            
            # 修改像素 - 减少它的值（因为正梯度表示增加该像素会增加真实类别的概率）
            x_adv[0, idx_c, idx_h, idx_w] = torch.clamp(
                x_adv[0, idx_c, idx_h, idx_w] - self.theta, 0, 1)
            
            # 标记像素已被修改
            modified[0, idx_c, idx_h, idx_w] = True
            modified_pixels += 1
            
            # 检查是否达到修改像素数量上限
            if modified_pixels >= max_pixels:
                break
                
        perturbation = x_adv - x
        return x_adv.detach(), perturbation.detach()
    
    def set_params(self, **kwargs):
        """设置攻击参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 