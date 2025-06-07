"""
配置文件
"""
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

# 创建必要的目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# matplotlib中文字体配置
def setup_matplotlib_chinese():
    """配置matplotlib支持中文显示"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    
    # 设置图像质量
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150

# 初始化matplotlib配置
setup_matplotlib_chinese()

# 数据集配置
DATASET_NAME = 'CIFAR10'
NUM_CLASSES = 10
IMAGE_SIZE = 32
BATCH_SIZE = 64

# 模型配置
MODEL_NAME = 'SimpleCNN'
LEARNING_RATE = 0.001
EPOCHS = 20

# 攻击参数默认值
ATTACK_PARAMS = {
    'FGSM': {
        'epsilon': 0.03,
        'clip_min': 0.0,
        'clip_max': 1.0
    },
    'PGD': {
        'epsilon': 0.03,
        'alpha': 0.01,
        'num_steps': 40,
        'random_start': True
    },
    'CW': {
        'c': 1e-4,
        'kappa': 0,
        'max_iterations': 1000,
        'learning_rate': 0.01
    },
    'JSMA': {
        'theta': 0.2,
        'gamma': 0.1,
        'max_iterations': 2000
    }
}

# 防御参数默认值
DEFENSE_PARAMS = {
    'adversarial_training': {
        'epsilon': 0.03,
        'learning_rate': 0.001,
        'epochs': 10
    }
}

# UI配置
WINDOW_TITLE = "对抗攻击与防御演示系统"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# 图像显示配置
FIGURE_SIZE = (8, 5)  # 减小默认图像大小
SUBPLOT_ADJUST = {'top': 0.9, 'bottom': 0.1, 'left': 0.1, 'right': 0.9} 