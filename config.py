"""
配置文件
"""
import os

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

# 创建必要的目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
        'theta': 1.0,
        'gamma': 0.1,
        'max_iterations': 2000
    }
}

# 防御参数默认值
DEFENSE_PARAMS = {
    'adversarial_training': {
        'epsilon': 0.03,
        'alpha': 0.01,
        'num_steps': 10
    },
    'adversarial_distillation': {
        'temperature': 100,
        'alpha': 0.7
    }
}

# UI配置
WINDOW_TITLE = "对抗攻击与防御演示系统"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800 