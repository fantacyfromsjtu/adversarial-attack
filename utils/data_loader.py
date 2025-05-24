"""
数据加载工具
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from config import DATA_DIR, BATCH_SIZE


def get_cifar10_loaders(batch_size=BATCH_SIZE, num_workers=4):
    """
    获取CIFAR-10数据加载器
    
    Args:
        batch_size: 批大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        classes: 类别名称列表
    """
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # CIFAR-10类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, test_loader, classes


def denormalize_cifar10(tensor):
    """
    反归一化CIFAR-10图像
    
    Args:
        tensor: 归一化的图像张量
    
    Returns:
        denormalized: 反归一化的图像张量
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    denormalized = tensor * std + mean
    return torch.clamp(denormalized, 0, 1) 