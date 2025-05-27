"""
对抗训练防御方法
"""
import torch
import torch.nn as nn
import torch.optim as optim
from attacks import FGSM


class AdversarialTraining:
    """
    对抗训练防御方法
    """
    def __init__(self, model, device, epsilon=0.03):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.attacker = FGSM(model, epsilon=epsilon)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 生成对抗样本
            self.model.train()  # 确保模型在训练模式
            adv_data, _ = self.attacker.generate(data, target)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 在干净样本和对抗样本上计算损失
            clean_output = self.model(data)
            adv_output = self.model(adv_data)
            
            clean_loss = self.criterion(clean_output, target)
            adv_loss = self.criterion(adv_output, target)
            
            # 混合损失
            loss = 0.5 * clean_loss + 0.5 * adv_loss
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = adv_output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def evaluate(self, test_loader, use_adv=True):
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            use_adv: 是否在对抗样本上评估
        
        Returns:
            accuracy: 准确率
            avg_loss: 平均损失
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if use_adv:
                    # 生成对抗样本
                    data, _ = self.attacker.generate(data, target)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss 