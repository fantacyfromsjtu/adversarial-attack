"""
数据加载工具
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import random
from PIL import Image
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


def get_single_cifar10_image(index=None):
    """
    快速获取单张CIFAR-10图像用于预览
    
    Args:
        index: 图像索引，如果为None则随机选择
    
    Returns:
        image: PIL图像对象 (32x32)
        label: 标签
        class_name: 类别名称
    """
    # CIFAR-10类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 创建最小的数据集对象，不下载
    try:
        # 尝试加载已存在的数据集
        dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=False,  # 使用测试集，数据量更小
            download=False,  # 不自动下载
            transform=None  # 不应用变换
        )
    except:
        # 如果数据集不存在，则下载
        dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=False,
            download=True,
            transform=None
        )
    
    # 选择图像索引
    if index is None:
        index = random.randint(0, len(dataset) - 1)
    else:
        index = min(index, len(dataset) - 1)
    
    # 获取单张图像
    image, label = dataset[index]
    
    return image, label, classes[label]


def get_cifar10_tensor_from_pil(pil_image, normalize=True, target_size=32):
    """
    将PIL图像转换为PyTorch张量
    
    Args:
        pil_image: PIL图像对象
        normalize: 是否应用CIFAR-10归一化
        target_size: 目标尺寸，用于模型输入（默认32x32）
    
    Returns:
        tensor: 图像张量 (1, 3, target_size, target_size)
    """
    # 如果需要，调整图像尺寸用于模型输入
    if target_size != pil_image.size[0]:
        # 使用双线性插值缩放到目标尺寸
        model_image = pil_image.resize((target_size, target_size), Image.BILINEAR)
    else:
        model_image = pil_image
    
    # 转换为张量
    transform_list = [transforms.ToTensor()]
    
    if normalize:
        transform_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    
    transform = transforms.Compose(transform_list)
    tensor = transform(model_image).unsqueeze(0)  # 添加batch维度
    
    return tensor 