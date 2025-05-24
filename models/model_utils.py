"""
模型相关的工具函数
"""
import torch
import os
from config import MODEL_DIR


def save_model(model, model_name, epoch=None):
    """
    保存模型
    
    Args:
        model: PyTorch模型
        model_name: 模型名称
        epoch: 训练轮数（可选）
    """
    if epoch is not None:
        filename = f"{model_name}_epoch_{epoch}.pth"
    else:
        filename = f"{model_name}.pth"
    
    filepath = os.path.join(MODEL_DIR, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'epoch': epoch
    }, filepath)
    print(f"模型已保存到: {filepath}")
    return filepath


def load_pretrained_model(model, filepath):
    """
    加载预训练模型
    
    Args:
        model: PyTorch模型实例
        filepath: 模型文件路径
    
    Returns:
        model: 加载了权重的模型
        epoch: 训练轮数（如果有）
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', None)
    
    print(f"成功加载模型: {filepath}")
    return model, epoch


def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    
    Args:
        model: PyTorch模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    Returns:
        accuracy: 准确率
        avg_loss: 平均损失
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss 