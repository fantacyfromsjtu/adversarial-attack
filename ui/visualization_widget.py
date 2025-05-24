"""
可视化模块UI
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from attacks import FGSM, PGD, CW, JSMA
from utils import get_cifar10_loaders, plot_attack_success_rate


class VisualizationWidget(QWidget):
    """可视化模块界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        
    def initUI(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建工具栏
        toolbar_layout = QHBoxLayout()
        
        # 可视化类型选择
        toolbar_layout.addWidget(QLabel("可视化类型:"))
        self.vis_combo = QComboBox()
        self.vis_combo.addItems([
            "攻击成功率对比",
            "扰动强度影响",
            "模型置信度分析"
        ])
        toolbar_layout.addWidget(self.vis_combo)
        
        # 生成按钮
        self.generate_btn = QPushButton("生成可视化")
        self.generate_btn.clicked.connect(self.generate_visualization)
        toolbar_layout.addWidget(self.generate_btn)
        
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)
        
        # 创建显示区域
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 信息显示
        self.info_label = QLabel("请选择可视化类型并生成")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
    def on_model_loaded(self):
        """模型加载完成的处理"""
        self.info_label.setText("模型已加载，可以生成可视化")
        
    def generate_visualization(self):
        """生成可视化"""
        if self.parent.get_model() is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
            
        vis_type = self.vis_combo.currentText()
        
        if vis_type == "攻击成功率对比":
            self.visualize_attack_comparison()
        elif vis_type == "扰动强度影响":
            self.visualize_epsilon_effect()
        elif vis_type == "模型置信度分析":
            self.visualize_confidence_analysis()
            
    def visualize_attack_comparison(self):
        """可视化不同攻击方法的成功率对比"""
        self.info_label.setText("正在计算不同攻击方法的成功率...")
        QApplication.processEvents()
        
        model = self.parent.get_model()
        device = self.parent.get_device()
        
        # 获取测试数据（小批量用于演示）
        _, test_loader, _ = get_cifar10_loaders(batch_size=100)
        
        # 只使用一个批次进行演示
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        
        # 测试不同攻击方法
        attack_methods = ['FGSM', 'PGD']
        success_rates = []
        
        for method in attack_methods:
            if method == 'FGSM':
                attacker = FGSM(model, epsilon=0.03)
            elif method == 'PGD':
                attacker = PGD(model, epsilon=0.03, alpha=0.01, num_steps=20)
                
            # 计算成功率
            success_count = 0
            for i in range(len(data)):
                x = data[i:i+1]
                y = target[i:i+1]
                
                # 原始预测
                with torch.no_grad():
                    orig_pred = model(x).argmax(1)
                
                # 生成对抗样本
                x_adv, _ = attacker.generate(x, y)
                
                # 对抗预测
                with torch.no_grad():
                    adv_pred = model(x_adv).argmax(1)
                
                if orig_pred != adv_pred:
                    success_count += 1
                    
            success_rate = (success_count / len(data)) * 100
            success_rates.append(success_rate)
            
        # 绘制结果
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        bars = ax.bar(attack_methods, success_rates, color=['#FF6B6B', '#4ECDC4'])
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        ax.set_ylabel('攻击成功率 (%)')
        ax.set_title('不同攻击方法的成功率比较 (ε=0.03)')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        self.canvas.draw()
        self.info_label.setText("攻击成功率对比生成完成")
        
    def visualize_epsilon_effect(self):
        """可视化扰动强度对攻击效果的影响"""
        self.info_label.setText("正在分析扰动强度的影响...")
        QApplication.processEvents()
        
        model = self.parent.get_model()
        device = self.parent.get_device()
        
        # 获取测试数据
        _, test_loader, _ = get_cifar10_loaders(batch_size=100)
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        
        # 测试不同的epsilon值
        epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]
        fgsm_success_rates = []
        pgd_success_rates = []
        
        for eps in epsilons:
            # FGSM
            if eps == 0.0:
                fgsm_success_rates.append(0)
                pgd_success_rates.append(0)
            else:
                # FGSM攻击
                fgsm_attacker = FGSM(model, epsilon=eps)
                fgsm_success = 0
                
                # PGD攻击
                pgd_attacker = PGD(model, epsilon=eps, alpha=eps/4, num_steps=10)
                pgd_success = 0
                
                for i in range(len(data)):
                    x = data[i:i+1]
                    y = target[i:i+1]
                    
                    with torch.no_grad():
                        orig_pred = model(x).argmax(1)
                    
                    # FGSM
                    x_adv_fgsm, _ = fgsm_attacker.generate(x, y)
                    with torch.no_grad():
                        adv_pred_fgsm = model(x_adv_fgsm).argmax(1)
                    if orig_pred != adv_pred_fgsm:
                        fgsm_success += 1
                    
                    # PGD
                    x_adv_pgd, _ = pgd_attacker.generate(x, y)
                    with torch.no_grad():
                        adv_pred_pgd = model(x_adv_pgd).argmax(1)
                    if orig_pred != adv_pred_pgd:
                        pgd_success += 1
                
                fgsm_success_rates.append((fgsm_success / len(data)) * 100)
                pgd_success_rates.append((pgd_success / len(data)) * 100)
        
        # 绘制结果
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(epsilons, fgsm_success_rates, 'o-', label='FGSM', linewidth=2, markersize=8)
        ax.plot(epsilons, pgd_success_rates, 's-', label='PGD', linewidth=2, markersize=8)
        
        ax.set_xlabel('扰动强度 (ε)')
        ax.set_ylabel('攻击成功率 (%)')
        ax.set_title('扰动强度对攻击成功率的影响')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-0.01, 0.11)
        ax.set_ylim(-5, 105)
        
        self.canvas.draw()
        self.info_label.setText("扰动强度影响分析完成")
        
    def visualize_confidence_analysis(self):
        """可视化模型置信度分析"""
        self.info_label.setText("正在分析模型置信度变化...")
        QApplication.processEvents()
        
        model = self.parent.get_model()
        device = self.parent.get_device()
        
        # 获取一个样本
        _, test_loader, classes = get_cifar10_loaders(batch_size=1)
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        
        # 使用FGSM生成不同强度的对抗样本
        epsilons = np.linspace(0, 0.1, 11)
        confidences = []
        predictions = []
        
        for eps in epsilons:
            if eps == 0:
                x_adv = data
            else:
                attacker = FGSM(model, epsilon=eps)
                x_adv, _ = attacker.generate(data, target)
            
            with torch.no_grad():
                output = model(x_adv)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = probs.max(1)
                
            confidences.append(confidence.item())
            predictions.append(pred.item())
        
        # 绘制结果
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        # 置信度变化
        ax1.plot(epsilons, confidences, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('扰动强度 (ε)')
        ax1.set_ylabel('模型置信度')
        ax1.set_title('模型置信度随扰动强度的变化')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 预测类别变化
        ax2.scatter(epsilons, predictions, c=predictions, cmap='tab10', s=100)
        ax2.set_xlabel('扰动强度 (ε)')
        ax2.set_ylabel('预测类别')
        ax2.set_title('预测类别随扰动强度的变化')
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(classes)
        ax2.grid(True, alpha=0.3)
        
        # 标记原始类别
        ax2.axhline(y=target.item(), color='red', linestyle='--', alpha=0.5, label='真实类别')
        ax2.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()
        self.info_label.setText("模型置信度分析完成") 