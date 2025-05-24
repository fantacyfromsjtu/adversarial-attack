"""
防御模块UI
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from defenses import AdversarialTraining, AdversarialDistillation
from utils import get_cifar10_loaders
from config import DEFENSE_PARAMS


class DefenseWidget(QWidget):
    """防御模块界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.defense_methods = {
            '对抗训练': AdversarialTraining,
            '对抗蒸馏': AdversarialDistillation
        }
        self.initUI()
        
    def initUI(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建工具栏
        toolbar_layout = QHBoxLayout()
        
        # 防御方法选择
        toolbar_layout.addWidget(QLabel("防御方法:"))
        self.defense_combo = QComboBox()
        self.defense_combo.addItems(list(self.defense_methods.keys()))
        toolbar_layout.addWidget(self.defense_combo)
        
        # 开始训练按钮
        self.train_btn = QPushButton("开始防御训练")
        self.train_btn.clicked.connect(self.start_training)
        toolbar_layout.addWidget(self.train_btn)
        
        # 评估按钮
        self.evaluate_btn = QPushButton("评估防御效果")
        self.evaluate_btn.clicked.connect(self.evaluate_defense)
        self.evaluate_btn.setEnabled(False)
        toolbar_layout.addWidget(self.evaluate_btn)
        
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)
        
        # 创建参数面板和显示区域
        content_layout = QHBoxLayout()
        
        # 左侧参数面板
        self.create_parameter_panel()
        content_layout.addWidget(self.param_widget, 1)
        
        # 右侧显示区域
        self.display_widget = QWidget()
        display_layout = QVBoxLayout(self.display_widget)
        
        # 创建matplotlib画布
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        display_layout.addWidget(self.canvas)
        
        # 训练信息
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        display_layout.addWidget(self.info_text)
        
        content_layout.addWidget(self.display_widget, 3)
        layout.addLayout(content_layout)
        
    def create_parameter_panel(self):
        """创建参数面板"""
        self.param_widget = QGroupBox("防御参数")
        param_layout = QFormLayout(self.param_widget)
        
        # 通用参数
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 50)
        self.epochs_spin.setValue(5)
        param_layout.addRow("训练轮数:", self.epochs_spin)
        
        # 对抗训练参数
        self.adv_epsilon_spin = QDoubleSpinBox()
        self.adv_epsilon_spin.setRange(0.0, 1.0)
        self.adv_epsilon_spin.setSingleStep(0.01)
        self.adv_epsilon_spin.setValue(DEFENSE_PARAMS['adversarial_training']['epsilon'])
        param_layout.addRow("扰动强度 (ε):", self.adv_epsilon_spin)
        
        # 攻击方法选择（用于对抗训练）
        self.attack_method_combo = QComboBox()
        self.attack_method_combo.addItems(['fgsm', 'pgd'])
        param_layout.addRow("攻击方法:", self.attack_method_combo)
        
    def on_model_loaded(self):
        """模型加载完成的处理"""
        self.info_text.append("模型已加载，可以开始防御训练。")
        
    def start_training(self):
        """开始防御训练"""
        if self.parent.get_model() is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
            
        defense_method = self.defense_combo.currentText()
        
        # 禁用按钮
        self.train_btn.setEnabled(False)
        
        # 创建进度对话框
        progress = QProgressDialog("正在进行防御训练...", "取消", 0, self.epochs_spin.value(), self)
        progress.setWindowModality(Qt.WindowModal)
        
        try:
            # 获取数据加载器
            train_loader, test_loader, _ = get_cifar10_loaders()
            
            # 获取模型
            model = self.parent.get_model()
            device = self.parent.get_device()
            
            # 清空信息显示
            self.info_text.clear()
            self.info_text.append(f"开始{defense_method}训练...")
            
            # 创建防御训练器
            if defense_method == '对抗训练':
                defender = AdversarialTraining(
                    model,
                    attack_method=self.attack_method_combo.currentText(),
                    epsilon=self.adv_epsilon_spin.value()
                )
                
                # 训练历史
                history = {'train_loss': [], 'train_acc': []}
                
                for epoch in range(self.epochs_spin.value()):
                    if progress.wasCanceled():
                        break
                        
                    avg_loss, avg_acc = defender.train_epoch(train_loader, epoch)
                    history['train_loss'].append(avg_loss)
                    history['train_acc'].append(avg_acc)
                    
                    self.info_text.append(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
                    progress.setValue(epoch + 1)
                    QApplication.processEvents()
                
                # 绘制训练曲线
                self.plot_training_curve(history)
                
            else:  # 对抗蒸馏
                self.info_text.append("对抗蒸馏功能需要教师模型，暂未实现完整功能。")
                
            self.info_text.append("训练完成！")
            self.evaluate_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败: {str(e)}")
            self.info_text.append(f"训练失败: {str(e)}")
            
        finally:
            self.train_btn.setEnabled(True)
            progress.close()
            
    def evaluate_defense(self):
        """评估防御效果"""
        if self.parent.get_model() is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
            
        self.info_text.append("\n开始评估防御效果...")
        
        try:
            # 获取测试数据
            _, test_loader, _ = get_cifar10_loaders()
            
            # 创建临时的攻击器来评估
            model = self.parent.get_model()
            device = self.parent.get_device()
            
            from attacks import FGSM, PGD
            
            # 测试不同强度的攻击
            epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]
            accuracies = []
            
            for eps in epsilons:
                if eps == 0.0:
                    # 干净样本的准确率
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data, target in test_loader:
                            data, target = data.to(device), target.to(device)
                            outputs = model(data)
                            _, predicted = outputs.max(1)
                            total += target.size(0)
                            correct += predicted.eq(target).sum().item()
                    acc = 100. * correct / total
                else:
                    # 对抗样本的准确率
                    attacker = FGSM(model, epsilon=eps)
                    correct = 0
                    total = 0
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        adv_data, _ = attacker.generate(data, target)
                        with torch.no_grad():
                            outputs = model(adv_data)
                            _, predicted = outputs.max(1)
                            total += target.size(0)
                            correct += predicted.eq(target).sum().item()
                    acc = 100. * correct / total
                    
                accuracies.append(acc)
                self.info_text.append(f"ε={eps:.2f}: 准确率={acc:.2f}%")
                
            # 绘制鲁棒性曲线
            self.plot_robustness_curve(epsilons, accuracies)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"评估失败: {str(e)}")
            self.info_text.append(f"评估失败: {str(e)}")
            
    def plot_training_curve(self, history):
        """绘制训练曲线"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='训练损失')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('损失')
        ax.set_title('训练损失曲线')
        ax.legend()
        ax.grid(True)
        
        self.canvas.draw()
        
    def plot_robustness_curve(self, epsilons, accuracies):
        """绘制鲁棒性曲线"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(epsilons, accuracies, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('扰动强度 (ε)')
        ax.set_ylabel('准确率 (%)')
        ax.set_title('模型鲁棒性评估')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 添加数值标签
        for eps, acc in zip(epsilons, accuracies):
            ax.annotate(f'{acc:.1f}%', xy=(eps, acc), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom')
        
        self.canvas.draw() 