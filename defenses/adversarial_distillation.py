"""
对抗蒸馏防御方法
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import SimpleCNN


class AdversarialDistillation:
    """
    对抗蒸馏防御方法
    """
    def __init__(self, student_model, device, temperature=100, alpha=0.7):
        self.student_model = student_model
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        # 创建教师模型（使用相同的架构）
        self.teacher_model = SimpleCNN().to(device)
        self.teacher_model.eval()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    
    def distillation_loss(self, student_outputs, teacher_outputs, temperature):
        """
        计算蒸馏损失
        """
        soft_student = F.log_softmax(student_outputs / temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / temperature, dim=1)
        loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
        return loss
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        """
        self.student_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 获取教师模型输出
            with torch.no_grad():
                teacher_outputs = self.teacher_model(data)
            
            # 获取学生模型输出
            student_outputs = self.student_model(data)
            
            # 计算损失
            # 硬标签损失
            hard_loss = self.criterion(student_outputs, target)
            # 软标签损失（蒸馏损失）
            soft_loss = self.distillation_loss(student_outputs, teacher_outputs, self.temperature)
            
            # 总损失
            loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def evaluate(self, test_loader):
        """
        评估学生模型
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            accuracy: 准确率
            avg_loss: 平均损失
        """
        self.student_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.student_model(data)
                loss = self.criterion(outputs, target)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss 