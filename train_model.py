"""
训练CNN模型的脚本
用于生成初始的预训练模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import SimpleCNN, save_model
from utils import get_cifar10_loaders
from config import LEARNING_RATE, EPOCHS


def train_model():
    """训练模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载CIFAR-10数据集...")
    train_loader, test_loader, classes = get_cifar10_loaders()
    
    # 创建模型
    print("创建CNN模型...")
    model = SimpleCNN(num_classes=10).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练模型
    print(f"开始训练，共{EPOCHS}个epoch...")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # 测试阶段
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Test Loss: {test_loss/len(test_loader):.3f}, '
              f'Test Acc: {test_acc:.2f}%')
        
        # 调整学习率
        scheduler.step()
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            save_model(model, 'SimpleCNN', epoch + 1)
    
    # 保存最终模型
    final_path = save_model(model, 'SimpleCNN')
    print(f"\n训练完成！最终模型保存在: {final_path}")
    print(f"最终测试准确率: {test_acc:.2f}%")


if __name__ == '__main__':
    train_model() 