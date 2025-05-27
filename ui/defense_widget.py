"""
防御模块UI
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from defenses import AdversarialTraining, AdversarialDistillation
from models import SimpleCNN
from utils import get_cifar10_loaders
from config import DEFENSE_PARAMS

# 确保matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DefenseWidget(QWidget):
    """防御模块界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.training_thread = None
        self.initUI()
        
    def initUI(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 防御方法选择
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("防御方法:"))
        
        self.defense_combo = QComboBox()
        self.defense_combo.addItems(['对抗训练', '对抗蒸馏'])
        method_layout.addWidget(self.defense_combo)
        
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
        # 参数设置
        param_group = QGroupBox("训练参数")
        param_layout = QFormLayout(param_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        param_layout.addRow("训练轮数:", self.epochs_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4)
        param_layout.addRow("学习率:", self.lr_spin)
        
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1.0)
        self.epsilon_spin.setSingleStep(0.01)
        self.epsilon_spin.setValue(DEFENSE_PARAMS['adversarial_training']['epsilon'])
        param_layout.addRow("扰动强度:", self.epsilon_spin)
        
        layout.addWidget(param_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_btn)
        
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.evaluate_btn = QPushButton("评估模型")
        self.evaluate_btn.clicked.connect(self.evaluate_model)
        self.evaluate_btn.setEnabled(False)
        button_layout.addWidget(self.evaluate_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 训练历史图表
        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMaximumHeight(300)
        layout.addWidget(self.canvas)
        
        # 信息显示
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.info_text)
        
        # 初始化图表
        self.init_plot()
        
    def init_plot(self):
        """初始化图表"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title("训练历史", fontsize=14)
        ax.text(0.5, 0.5, "开始训练后将显示训练历史", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.figure.tight_layout()
        self.canvas.draw()
        
    def on_model_loaded(self):
        """模型加载完成的处理"""
        self.evaluate_btn.setEnabled(True)
        self.info_text.append("模型已加载，可以开始训练或评估")
        
    def start_training(self):
        """开始训练"""
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "警告", "训练正在进行中！")
            return
            
        # 获取参数
        defense_method = self.defense_combo.currentText()
        epochs = self.epochs_spin.value()
        learning_rate = self.lr_spin.value()
        epsilon = self.epsilon_spin.value()
        
        # 创建训练线程
        self.training_thread = TrainingThread(defense_method, epochs, learning_rate, epsilon)
        self.training_thread.progress_updated.connect(self.update_progress)
        self.training_thread.epoch_completed.connect(self.update_plot)
        self.training_thread.training_completed.connect(self.training_finished)
        self.training_thread.info_updated.connect(self.update_info)
        
        # 更新UI状态
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.info_text.clear()
        
        # 开始训练
        self.training_thread.start()
        
    def stop_training(self):
        """停止训练"""
        if self.training_thread:
            self.training_thread.stop()
            self.training_finished()
            
    def training_finished(self):
        """训练完成"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.evaluate_btn.setEnabled(True)
        self.info_text.append("训练完成！")
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def update_info(self, text):
        """更新信息显示"""
        self.info_text.append(text)
        
    def update_plot(self, history):
        """更新训练历史图表"""
        self.figure.clear()
        
        if len(history['train_loss']) > 0:
            ax1 = self.figure.add_subplot(121)
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失')
            if 'val_loss' in history and len(history['val_loss']) > 0:
                ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('损失')
            ax1.set_title('训练损失')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = self.figure.add_subplot(122)
            ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
            if 'val_acc' in history and len(history['val_acc']) > 0:
                ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('准确率 (%)')
            ax2.set_title('训练准确率')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def evaluate_model(self):
        """评估模型"""
        model = self.parent.get_model()
        if model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
            
        device = self.parent.get_device()
        
        try:
            # 获取测试数据
            _, test_loader, classes = get_cifar10_loaders()
            
            model.eval()
            correct = 0
            total = 0
            
            self.info_text.append("正在评估模型...")
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = outputs.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    if total >= 1000:  # 限制测试样本数量
                        break
            
            accuracy = 100. * correct / total
            self.info_text.append(f"模型准确率: {accuracy:.2f}% ({correct}/{total})")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"评估失败: {str(e)}")


class TrainingThread(QThread):
    """训练线程"""
    progress_updated = pyqtSignal(int)
    epoch_completed = pyqtSignal(dict)
    training_completed = pyqtSignal()
    info_updated = pyqtSignal(str)
    
    def __init__(self, defense_method, epochs, learning_rate, epsilon):
        super().__init__()
        self.defense_method = defense_method
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.model = None
        self.is_stopped = False
        
    def stop(self):
        """停止训练"""
        self.is_stopped = True
        
    def run(self):
        """运行训练"""
        try:
            self.info_updated.emit(f"开始{self.defense_method}训练...")
            
            # 获取数据
            train_loader, test_loader, classes = get_cifar10_loaders()
            
            # 创建模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = SimpleCNN(num_classes=len(classes)).to(device)
            
            # 创建防御训练器
            if self.defense_method == '对抗训练':
                trainer = AdversarialTraining(self.model, device)
            else:  # 对抗蒸馏
                trainer = AdversarialDistillation(self.model, device)
            
            # 训练历史
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            
            for epoch in range(self.epochs):
                if self.is_stopped:
                    break
                    
                # 训练一个epoch
                train_loss, train_acc = trainer.train_epoch(train_loader, epoch + 1)
                
                # 验证
                val_loss, val_acc = self.validate(test_loader, device)
                
                # 更新历史
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # 发送信号
                progress = int((epoch + 1) / self.epochs * 100)
                self.progress_updated.emit(progress)
                self.epoch_completed.emit(history)
                self.info_updated.emit(f"Epoch {epoch+1}/{self.epochs}: "
                                     f"训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}%, "
                                     f"验证损失={val_loss:.4f}, 验证准确率={val_acc:.2f}%")
            
            if not self.is_stopped:
                self.training_completed.emit()
                
        except Exception as e:
            self.info_updated.emit(f"训练出错: {str(e)}")
            
    def validate(self, test_loader, device):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                loss = criterion(outputs, target)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if total >= 1000:  # 限制验证样本数量
                    break
        
        avg_loss = total_loss / min(len(test_loader), 1000 // test_loader.batch_size)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc 