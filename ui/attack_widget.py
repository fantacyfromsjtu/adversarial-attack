"""
攻击模块UI
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from attacks import FGSM, PGD, CW, JSMA
from utils import get_cifar10_loaders, visualize_attack_results
from config import ATTACK_PARAMS


class AttackWidget(QWidget):
    """攻击模块界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_image = None
        self.current_label = None
        self.attack_methods = {
            'FGSM': FGSM,
            'PGD': PGD,
            'CW': CW,
            'JSMA': JSMA
        }
        self.initUI()
        
    def initUI(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建工具栏
        toolbar_layout = QHBoxLayout()
        
        # 攻击方法选择
        toolbar_layout.addWidget(QLabel("攻击方法:"))
        self.attack_combo = QComboBox()
        self.attack_combo.addItems(list(self.attack_methods.keys()))
        self.attack_combo.currentTextChanged.connect(self.on_attack_method_changed)
        toolbar_layout.addWidget(self.attack_combo)
        
        # 加载图像按钮
        self.load_image_btn = QPushButton("加载测试图像")
        self.load_image_btn.clicked.connect(self.load_test_image)
        toolbar_layout.addWidget(self.load_image_btn)
        
        # 执行攻击按钮
        self.attack_btn = QPushButton("执行攻击")
        self.attack_btn.clicked.connect(self.execute_attack)
        self.attack_btn.setEnabled(False)
        toolbar_layout.addWidget(self.attack_btn)
        
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
        
        # 结果信息
        self.result_label = QLabel("请加载图像并执行攻击")
        self.result_label.setAlignment(Qt.AlignCenter)
        display_layout.addWidget(self.result_label)
        
        content_layout.addWidget(self.display_widget, 3)
        layout.addLayout(content_layout)
        
    def create_parameter_panel(self):
        """创建参数面板"""
        self.param_widget = QGroupBox("攻击参数")
        self.param_layout = QFormLayout(self.param_widget)
        
        # FGSM参数
        self.fgsm_params = {
            'epsilon': QDoubleSpinBox()
        }
        self.fgsm_params['epsilon'].setRange(0.0, 1.0)
        self.fgsm_params['epsilon'].setSingleStep(0.01)
        self.fgsm_params['epsilon'].setValue(ATTACK_PARAMS['FGSM']['epsilon'])
        
        # PGD参数
        self.pgd_params = {
            'epsilon': QDoubleSpinBox(),
            'alpha': QDoubleSpinBox(),
            'num_steps': QSpinBox()
        }
        self.pgd_params['epsilon'].setRange(0.0, 1.0)
        self.pgd_params['epsilon'].setSingleStep(0.01)
        self.pgd_params['epsilon'].setValue(ATTACK_PARAMS['PGD']['epsilon'])
        
        self.pgd_params['alpha'].setRange(0.0, 1.0)
        self.pgd_params['alpha'].setSingleStep(0.001)
        self.pgd_params['alpha'].setValue(ATTACK_PARAMS['PGD']['alpha'])
        
        self.pgd_params['num_steps'].setRange(1, 100)
        self.pgd_params['num_steps'].setValue(ATTACK_PARAMS['PGD']['num_steps'])
        
        # 初始显示FGSM参数
        self.update_parameter_panel('FGSM')
        
    def update_parameter_panel(self, attack_method):
        """更新参数面板"""
        # 清除现有参数
        while self.param_layout.count():
            child = self.param_layout.takeAt(0)
            if child.widget():
                child.widget().hide()
        
        # 显示对应攻击方法的参数
        if attack_method == 'FGSM':
            self.param_layout.addRow("扰动强度 (ε):", self.fgsm_params['epsilon'])
            self.fgsm_params['epsilon'].show()
            
        elif attack_method == 'PGD':
            self.param_layout.addRow("扰动强度 (ε):", self.pgd_params['epsilon'])
            self.param_layout.addRow("步长 (α):", self.pgd_params['alpha'])
            self.param_layout.addRow("迭代次数:", self.pgd_params['num_steps'])
            for widget in self.pgd_params.values():
                widget.show()
                
        # 简化CW和JSMA，使用默认参数
        elif attack_method in ['CW', 'JSMA']:
            info_label = QLabel(f"使用{attack_method}默认参数")
            self.param_layout.addRow(info_label)
            
    def on_attack_method_changed(self, method):
        """攻击方法改变时的处理"""
        self.update_parameter_panel(method)
        
    def on_model_loaded(self):
        """模型加载完成的处理"""
        self.attack_btn.setEnabled(self.current_image is not None)
        
    def load_test_image(self):
        """加载测试图像"""
        if self.parent.get_model() is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
            
        # 获取测试数据
        _, test_loader, self.classes = get_cifar10_loaders(batch_size=1)
        
        # 随机选择一张图像
        data_iter = iter(test_loader)
        self.current_image, self.current_label = next(data_iter)
        
        # 显示原始图像
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 反归一化并显示
        img = self.current_image.squeeze().permute(1, 2, 0).numpy()
        img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"原始图像 - 类别: {self.classes[self.current_label.item()]}")
        ax.axis('off')
        
        self.canvas.draw()
        self.attack_btn.setEnabled(True)
        self.result_label.setText("图像已加载，可以执行攻击")
        
    def execute_attack(self):
        """执行攻击"""
        if self.current_image is None:
            return
            
        model = self.parent.get_model()
        device = self.parent.get_device()
        
        if model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
            
        # 获取当前攻击方法
        attack_method = self.attack_combo.currentText()
        
        # 准备数据
        x = self.current_image.to(device)
        y = self.current_label.to(device)
        
        # 创建攻击器
        if attack_method == 'FGSM':
            attacker = FGSM(model, epsilon=self.fgsm_params['epsilon'].value())
        elif attack_method == 'PGD':
            attacker = PGD(model, 
                          epsilon=self.pgd_params['epsilon'].value(),
                          alpha=self.pgd_params['alpha'].value(),
                          num_steps=self.pgd_params['num_steps'].value())
        else:
            attacker = self.attack_methods[attack_method](model)
            
        # 执行攻击
        try:
            x_adv, perturbation = attacker.generate(x, y)
            
            # 获取预测结果
            with torch.no_grad():
                original_output = model(x)
                adversarial_output = model(x_adv)
                
            original_pred = original_output.argmax(1).item()
            adversarial_pred = adversarial_output.argmax(1).item()
            
            # 可视化结果
            self.figure.clear()
            fig = visualize_attack_results(
                x, x_adv, perturbation,
                original_pred, adversarial_pred,
                self.classes
            )
            
            # 将matplotlib图形复制到我们的画布
            for ax in fig.axes:
                self.figure.add_subplot(ax)
            
            self.canvas.draw()
            
            # 更新结果信息
            success = original_pred != adversarial_pred
            self.result_label.setText(
                f"攻击{'成功' if success else '失败'}！"
                f" 原始预测: {self.classes[original_pred]}，"
                f" 对抗预测: {self.classes[adversarial_pred]}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"攻击执行失败: {str(e)}")
            print(f"错误详情: {e}")  # 调试用 