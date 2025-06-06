"""
C&W (Carlini & Wagner) 攻击标准实现
"""
import torch
import torch.nn as nn
import torch.optim as optim


class CW:
    """
    C&W L2攻击方法（标准实现）
    
    参数:
        model: 目标模型
        c: 平衡参数，控制对抗性和扰动大小
        kappa: 置信度参数
        steps: 优化步数
        lr: 学习率
    """
    def __init__(self, model, c=1.0, kappa=0, steps=1000, lr=0.01):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
    
    def generate(self, x, y):
        """
        生成对抗样本
        
        参数:
            x: 原始输入图像
            y: 真实标签
        
        返回:
            x_adv: 对抗样本
            perturbation: 扰动
        """
        # 自动检测并使用GPU（如果可用）
        device = next(self.model.parameters()).device
        x = x.to(device)
        y = y.to(device)
        
        # 初始化
        batch_size = x.shape[0]
        
        # 使用w变量表示对抗样本，通过tanh函数保证值域在[-1,1]
        w = self.inverse_tanh_space(x).detach()
        w.requires_grad = True
        
        # 设置优化器
        optimizer = optim.Adam([w], lr=self.lr)
        
        # 目标类（针对无目标攻击，使用原始标签）
        y_target = y
        
        # 最佳对抗样本
        best_adv = x.clone()
        best_l2 = 1e10 * torch.ones(batch_size, device=device)
        prev_cost = 1e10
        
        # 优化循环
        for step in range(self.steps):
            # 计算对抗样本
            x_adv = self.tanh_space(w)
            
            # 计算扰动L2范数
            delta = x_adv - x
            l2_dist = torch.sum(delta.pow(2), dim=[1,2,3]).sqrt()
            
            # 模型预测
            outputs = self.model(x_adv)
            
            # 计算目标函数值（对于所有类别）
            real = outputs.gather(1, y_target.unsqueeze(1)).squeeze(1)
            other, _ = torch.max(outputs * (1 - torch.eye(outputs.shape[1], device=device)[y_target]), dim=1)
            
            # C&W损失: max(0, other - real + kappa)
            f_loss = torch.clamp(other - real + self.kappa, min=0)
            
            # 总损失
            cost = self.c * f_loss + l2_dist
            
            # 更新最佳对抗样本
            is_success = (outputs.argmax(1) != y_target)
            is_better_adv = (is_success & (l2_dist < best_l2))
            
            if is_better_adv.any():
                idx = is_better_adv.nonzero().squeeze()
                best_l2[idx] = l2_dist[idx]
                best_adv[idx] = x_adv[idx].clone()
            
            # 提前停止条件
            if cost.item() > prev_cost * 0.99999:
                break
            prev_cost = cost.item()
            
            # 反向传播和优化
            optimizer.zero_grad()
            cost.sum().backward()
            optimizer.step()
            
        # 返回最佳对抗样本
        perturbation = best_adv - x
        return best_adv.detach(), perturbation.detach()
    
    def tanh_space(self, w):
        """
        将变量从w空间转换到x空间，保证x在[0,1]范围内
        """
        return 0.5 * (torch.tanh(w) + 1)
    
    def inverse_tanh_space(self, x):
        """
        将变量从x空间转换到w空间
        """
        # 避免出现tanh的边界值
        x = torch.clamp(x, 1e-7, 1 - 1e-7)
        return 0.5 * torch.log((1 + x) / (1 - x))
    
    def set_params(self, **kwargs):
        """设置攻击参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 