"""
攻击与分析模块UI
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from attacks import FGSM, PGD, CW, JSMA
from utils import get_cifar10_loaders, get_single_cifar10_image, get_cifar10_tensor_from_pil
from config import ATTACK_PARAMS

# 确保matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AttackWidget(QWidget):
    """攻击与分析模块界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_image = None  # 用于模型推理的32x32张量
        self.current_label = None
        self.current_display_image = None  # 用于显示的128x128 PIL图像
        self.classes = None
        self.attack_methods = {'FGSM': FGSM, 'PGD': PGD, 'CW': CW, 'JSMA': JSMA}
        self.initUI()
        
    def initUI(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 单张图像攻击标签页
        self.single_attack_widget = self.create_single_attack_tab()
        self.tab_widget.addTab(self.single_attack_widget, "单张图像攻击")
        
        # 批量攻击分析标签页
        self.batch_analysis_widget = self.create_batch_analysis_tab()
        self.tab_widget.addTab(self.batch_analysis_widget, "批量攻击分析")
        
    def create_single_attack_tab(self):
        """创建单张图像攻击标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 工具栏
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
        
        # 内容区域
        content_layout = QHBoxLayout()
        
        # 左侧参数面板
        self.create_parameter_panel()
        content_layout.addWidget(self.param_widget, 1)
        
        # 右侧显示区域
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        
        # 创建matplotlib画布
        self.single_figure = Figure(figsize=(2, 1.5), dpi=100)
        self.single_canvas = FigureCanvas(self.single_figure)
        self.single_canvas.setMaximumHeight(150)
        display_layout.addWidget(self.single_canvas)
        
        # 结果信息
        self.result_label = QLabel("请加载图像并执行攻击")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 12px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        display_layout.addWidget(self.result_label)
        
        content_layout.addWidget(display_widget, 3)
        layout.addLayout(content_layout)
        
        return widget 

    def create_batch_analysis_tab(self):
        """创建批量攻击分析标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        # 分析类型选择
        toolbar_layout.addWidget(QLabel("分析类型:"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(['攻击成功率对比', '扰动强度影响'])
        toolbar_layout.addWidget(self.analysis_combo)
        
        # 开始分析按钮
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.clicked.connect(self.start_batch_analysis)
        self.analyze_btn.setEnabled(False)
        toolbar_layout.addWidget(self.analyze_btn)
        
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)
        
        # 内容区域
        content_layout = QHBoxLayout()
        
        # 左侧参数面板
        self.create_analysis_parameter_panel()
        content_layout.addWidget(self.analysis_param_widget, 1)
        
        # 右侧显示区域
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        
        # 创建matplotlib画布
        self.batch_figure = Figure(figsize=(10, 6), dpi=100)
        self.batch_canvas = FigureCanvas(self.batch_figure)
        self.batch_canvas.setMaximumHeight(450)
        display_layout.addWidget(self.batch_canvas)
        
        # 添加进度条
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False) # 初始隐藏
        display_layout.addWidget(self.batch_progress_bar)
        
        # 分析结果文本
        self.analysis_text = QTextEdit()
        self.analysis_text.setMaximumHeight(120)
        self.analysis_text.setStyleSheet("font-size: 11px;")
        self.analysis_text.setPlaceholderText("分析结果将在这里显示...")
        display_layout.addWidget(self.analysis_text)
        
        content_layout.addWidget(display_widget, 3)
        layout.addLayout(content_layout)
        
        # 初始化图表
        self.init_batch_plot()
        
        return widget
        
    def create_parameter_panel(self):
        """创建攻击参数面板"""
        self.param_widget = QGroupBox("攻击参数")
        self.param_layout = QFormLayout(self.param_widget)
        
        # FGSM参数
        self.fgsm_params = {'epsilon': QDoubleSpinBox()}
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
        
        # CW参数
        self.cw_params = {
            'c': QDoubleSpinBox(),
            'kappa': QDoubleSpinBox(),
            'steps': QSpinBox(),
            'lr': QDoubleSpinBox()
        }
        self.cw_params['c'].setRange(1e-5, 10.0)
        self.cw_params['c'].setSingleStep(0.1)
        self.cw_params['c'].setDecimals(5)
        self.cw_params['c'].setValue(ATTACK_PARAMS['CW']['c'] if 'c' in ATTACK_PARAMS['CW'] else 1.0)
        
        self.cw_params['kappa'].setRange(0.0, 100.0)
        self.cw_params['kappa'].setSingleStep(1.0)
        self.cw_params['kappa'].setValue(ATTACK_PARAMS['CW']['kappa'] if 'kappa' in ATTACK_PARAMS['CW'] else 0.0)
        
        self.cw_params['steps'].setRange(10, 2000)
        self.cw_params['steps'].setValue(ATTACK_PARAMS['CW']['steps'] if 'steps' in ATTACK_PARAMS['CW'] else 1000)
        
        self.cw_params['lr'].setRange(0.001, 0.1)
        self.cw_params['lr'].setSingleStep(0.001)
        self.cw_params['lr'].setDecimals(4)
        self.cw_params['lr'].setValue(ATTACK_PARAMS['CW']['lr'] if 'lr' in ATTACK_PARAMS['CW'] else 0.01)
        
        # JSMA参数
        self.jsma_params = {
            'theta': QDoubleSpinBox(),
            'gamma': QDoubleSpinBox(),
            'max_iter': QSpinBox()
        }
        self.jsma_params['theta'].setRange(0.01, 1.0)
        self.jsma_params['theta'].setSingleStep(0.01)
        self.jsma_params['theta'].setValue(ATTACK_PARAMS['JSMA']['theta'] if 'theta' in ATTACK_PARAMS['JSMA'] else 0.2)
        
        self.jsma_params['gamma'].setRange(0.01, 1.0)
        self.jsma_params['gamma'].setSingleStep(0.01)
        self.jsma_params['gamma'].setValue(ATTACK_PARAMS['JSMA']['gamma'] if 'gamma' in ATTACK_PARAMS['JSMA'] else 1.0)
        
        self.jsma_params['max_iter'].setRange(10, 5000)
        self.jsma_params['max_iter'].setValue(ATTACK_PARAMS['JSMA']['max_iter'] if 'max_iter' in ATTACK_PARAMS['JSMA'] else 100)
        
        # 初始显示FGSM参数
        self.update_parameter_panel('FGSM')
        
    def _create_asr_common_params_group(self):
        group = QGroupBox("对比通用参数")
        layout = QFormLayout(group)
        self.asr_sample_size_spin = QSpinBox()
        self.asr_sample_size_spin.setRange(10, 1000)
        self.asr_sample_size_spin.setValue(100)
        layout.addRow("样本数量：", self.asr_sample_size_spin)
        group.setVisible(False)
        return group

    def _create_asr_fgsm_param_group(self):
        group = QGroupBox("FGSM 参数 (对比)")
        layout = QFormLayout(group)
        self.asr_fgsm_epsilon = QDoubleSpinBox()
        self.asr_fgsm_epsilon.setRange(0.0, 1.0)
        self.asr_fgsm_epsilon.setSingleStep(0.01)
        self.asr_fgsm_epsilon.setValue(ATTACK_PARAMS['FGSM']['epsilon'])
        layout.addRow("扰动强度 (ε):", self.asr_fgsm_epsilon)
        group.setVisible(False)
        return group

    def _create_asr_pgd_param_group(self):
        group = QGroupBox("PGD 参数 (对比)")
        layout = QFormLayout(group)
        self.asr_pgd_epsilon = QDoubleSpinBox()
        self.asr_pgd_epsilon.setRange(0.0, 1.0)
        self.asr_pgd_epsilon.setSingleStep(0.01)
        self.asr_pgd_epsilon.setValue(ATTACK_PARAMS['PGD']['epsilon'])
        layout.addRow("扰动强度 (ε):", self.asr_pgd_epsilon)

        self.asr_pgd_alpha = QDoubleSpinBox()
        self.asr_pgd_alpha.setRange(0.0, 1.0)
        self.asr_pgd_alpha.setSingleStep(0.001)
        self.asr_pgd_alpha.setValue(ATTACK_PARAMS['PGD']['alpha'])
        layout.addRow("步长 (α):", self.asr_pgd_alpha)

        self.asr_pgd_num_steps = QSpinBox()
        self.asr_pgd_num_steps.setRange(1, 100)
        self.asr_pgd_num_steps.setValue(ATTACK_PARAMS['PGD']['num_steps'])
        layout.addRow("迭代次数:", self.asr_pgd_num_steps)
        group.setVisible(False)
        return group

    def _create_asr_cw_param_group(self):
        group = QGroupBox("CW 参数 (对比)")
        layout = QFormLayout(group)
        self.asr_cw_c = QDoubleSpinBox()
        self.asr_cw_c.setRange(1e-5, 1.0)
        self.asr_cw_c.setSingleStep(1e-5)
        self.asr_cw_c.setDecimals(5)
        self.asr_cw_c.setValue(ATTACK_PARAMS['CW']['c'])
        layout.addRow("权衡系数 (c):", self.asr_cw_c)

        self.asr_cw_max_iterations = QSpinBox()
        self.asr_cw_max_iterations.setRange(10, 2000)
        self.asr_cw_max_iterations.setValue(ATTACK_PARAMS['CW']['max_iterations'])
        layout.addRow("最大迭代:", self.asr_cw_max_iterations)
        group.setVisible(False)
        return group

    def _create_asr_jsma_param_group(self):
        group = QGroupBox("JSMA 参数 (对比)")
        layout = QFormLayout(group)
        self.asr_jsma_theta = QDoubleSpinBox()
        self.asr_jsma_theta.setRange(0.01, 1.0)
        self.asr_jsma_theta.setSingleStep(0.01)
        self.asr_jsma_theta.setValue(ATTACK_PARAMS['JSMA']['theta'] if 'theta' in ATTACK_PARAMS['JSMA'] else 0.2)
        layout.addRow("修改幅度 (θ):", self.asr_jsma_theta)

        self.asr_jsma_gamma = QDoubleSpinBox()
        self.asr_jsma_gamma.setRange(0.01, 1.0)
        self.asr_jsma_gamma.setSingleStep(0.01)
        self.asr_jsma_gamma.setValue(ATTACK_PARAMS['JSMA']['gamma'] if 'gamma' in ATTACK_PARAMS['JSMA'] else 1.0)
        layout.addRow("修改像素比例 (γ):", self.asr_jsma_gamma)

        self.asr_jsma_max_iterations = QSpinBox()
        self.asr_jsma_max_iterations.setRange(10, 5000)
        self.asr_jsma_max_iterations.setValue(ATTACK_PARAMS['JSMA']['max_iter'] if 'max_iter' in ATTACK_PARAMS['JSMA'] else 100)
        layout.addRow("最大迭代:", self.asr_jsma_max_iterations)
        
        group.setVisible(False)
        return group

    def create_analysis_parameter_panel(self):
        """创建分析参数面板 - 重构以支持动态参数"""
        self.analysis_param_widget = QGroupBox("分析参数")
        main_layout = QVBoxLayout(self.analysis_param_widget) # Main layout for the groupbox

        # Common parameters / Epsilon effect parameters
        self.epsilon_effect_params_group = QGroupBox("扰动强度影响参数")
        epsilon_layout = QFormLayout(self.epsilon_effect_params_group)
        
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(10, 1000)
        self.sample_size_spin.setValue(100)
        epsilon_layout.addRow("样本数量:", self.sample_size_spin) # Sample size is common
        
        self.epsilon_min = QDoubleSpinBox()
        self.epsilon_min.setRange(0.0, 0.5)
        self.epsilon_min.setSingleStep(0.01)
        self.epsilon_min.setValue(0.01)
        epsilon_layout.addRow("最小扰动 (ε):", self.epsilon_min)
        
        self.epsilon_max = QDoubleSpinBox()
        self.epsilon_max.setRange(0.0, 0.5)
        self.epsilon_max.setSingleStep(0.01)
        self.epsilon_max.setValue(0.1)
        epsilon_layout.addRow("最大扰动 (ε):", self.epsilon_max)
        
        self.epsilon_steps = QSpinBox()
        self.epsilon_steps.setRange(3, 20)
        self.epsilon_steps.setValue(6)
        epsilon_layout.addRow("采样步数:", self.epsilon_steps)
        main_layout.addWidget(self.epsilon_effect_params_group)

        # ASR Mode - Common Parameters (like its own sample size)
        self.asr_common_params_group = self._create_asr_common_params_group()
        main_layout.addWidget(self.asr_common_params_group)

        # Attack method selection
        self.attack_method_selection_group = QGroupBox("选择攻击方法 (用于对比)")
        checkbox_layout = QVBoxLayout(self.attack_method_selection_group) # Changed to QVBoxLayout for checkboxes

        self.fgsm_check = QCheckBox("FGSM")
        self.fgsm_check.setChecked(True)
        checkbox_layout.addWidget(self.fgsm_check)
        
        self.pgd_check = QCheckBox("PGD")
        self.pgd_check.setChecked(True)
        checkbox_layout.addWidget(self.pgd_check)
        
        self.cw_check = QCheckBox("CW")
        self.cw_check.setChecked(False)
        checkbox_layout.addWidget(self.cw_check)
        
        self.jsma_check = QCheckBox("JSMA")
        self.jsma_check.setChecked(False)
        checkbox_layout.addWidget(self.jsma_check)
        main_layout.addWidget(self.attack_method_selection_group)

        # Parameter groups for "Attack Success Rate" (ASR) mode
        self.asr_fgsm_params_group = self._create_asr_fgsm_param_group()
        main_layout.addWidget(self.asr_fgsm_params_group)
        self.asr_pgd_params_group = self._create_asr_pgd_param_group()
        main_layout.addWidget(self.asr_pgd_params_group)
        self.asr_cw_params_group = self._create_asr_cw_param_group()
        main_layout.addWidget(self.asr_cw_params_group)
        self.asr_jsma_params_group = self._create_asr_jsma_param_group()
        main_layout.addWidget(self.asr_jsma_params_group)
        
        main_layout.addStretch() # Add stretch to push elements to the top

        # Connect signals for dynamic UI updates
        self.analysis_combo.currentTextChanged.connect(self._on_analysis_type_changed)
        self.fgsm_check.toggled.connect(self._update_asr_param_widgets_visibility)
        self.pgd_check.toggled.connect(self._update_asr_param_widgets_visibility)
        self.cw_check.toggled.connect(self._update_asr_param_widgets_visibility)
        self.jsma_check.toggled.connect(self._update_asr_param_widgets_visibility)

        self._on_analysis_type_changed(self.analysis_combo.currentText()) # Initial UI setup

    def _on_analysis_type_changed(self, analysis_type):
        is_asr_mode = (analysis_type == '攻击成功率对比')
        is_epsilon_mode = (analysis_type == '扰动强度影响')

        # Visibility for Epsilon Effect parameters (includes its own sample_size_spin)
        self.epsilon_effect_params_group.setVisible(is_epsilon_mode)

        # Visibility for ASR Common Parameters (e.g. asr_sample_size_spin)
        self.asr_common_params_group.setVisible(is_asr_mode)

        # Visibility and enabled state for checkboxes
        self.attack_method_selection_group.setVisible(True) # Checkboxes always visible, but some might be disabled

        self.fgsm_check.setEnabled(True) # FGSM always available
        self.pgd_check.setEnabled(True)  # PGD always available
        
        if is_epsilon_mode:
            self.cw_check.setEnabled(False)
            self.cw_check.setChecked(False) # Uncheck if disabled
            self.jsma_check.setEnabled(False)
            self.jsma_check.setChecked(False) # Uncheck if disabled
        else: # ASR mode or other modes
            self.cw_check.setEnabled(True)
            self.jsma_check.setEnabled(True)
            
        self._update_asr_param_widgets_visibility()

    def _update_asr_param_widgets_visibility(self):
        # This method controls visibility of ASR parameter groups
        # Only show ASR param groups if in ASR mode
        analysis_type = self.analysis_combo.currentText()
        is_asr_mode = (analysis_type == '攻击成功率对比')

        self.asr_fgsm_params_group.setVisible(is_asr_mode and self.fgsm_check.isChecked())
        self.asr_pgd_params_group.setVisible(is_asr_mode and self.pgd_check.isChecked())
        self.asr_cw_params_group.setVisible(is_asr_mode and self.cw_check.isChecked())
        self.asr_jsma_params_group.setVisible(is_asr_mode and self.jsma_check.isChecked())
        
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
        elif attack_method == 'CW':
            self.param_layout.addRow("权衡系数 (c):", self.cw_params['c'])
            self.param_layout.addRow("置信度 (κ):", self.cw_params['kappa'])
            self.param_layout.addRow("优化步数:", self.cw_params['steps'])
            self.param_layout.addRow("学习率:", self.cw_params['lr'])
            for widget in self.cw_params.values():
                widget.show()
        elif attack_method == 'JSMA':
            self.param_layout.addRow("修改幅度 (θ):", self.jsma_params['theta'])
            self.param_layout.addRow("修改像素比例 (γ):", self.jsma_params['gamma'])
            self.param_layout.addRow("最大迭代:", self.jsma_params['max_iter'])
            for widget in self.jsma_params.values():
                widget.show()
        else:
            info_label = QLabel(f"该攻击方法使用默认参数或无用户可调参数。")
            self.param_layout.addRow(info_label)
            
    def on_attack_method_changed(self, method):
        """攻击方法改变时的处理"""
        self.update_parameter_panel(method)
        
    def on_model_loaded(self):
        """模型加载完成的处理"""
        self.attack_btn.setEnabled(self.current_image is not None)
        self.analyze_btn.setEnabled(True)
        
    def load_test_image(self):
        """加载测试图像"""
        if self.parent.get_model() is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
            
        try:
            # 快速获取单张图像（128x128）
            pil_image, label, class_name = get_single_cifar10_image()
            
            # 保存显示用的128x128图像
            self.current_display_image = pil_image
            
            # 转换为32x32张量用于模型推理
            self.current_image = get_cifar10_tensor_from_pil(pil_image, normalize=True, target_size=32)
            self.current_label = torch.tensor([label])
            
            # 设置类别名称
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
            
            # 显示原始图像（128x128）
            self.single_figure.clear()
            ax = self.single_figure.add_subplot(111)
            
            # 直接显示PIL图像
            ax.imshow(np.array(pil_image))
            ax.set_title(f"原始图像 (32x32) - 类别: {class_name}", fontsize=14)
            ax.axis('off')
            
            self.single_figure.tight_layout()
            self.single_canvas.draw()
            self.attack_btn.setEnabled(True)
            self.result_label.setText("图像已加载，可以执行攻击")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
            print(f"加载图像错误: {e}")  # 调试信息
        
    def execute_attack(self):
        """执行单张图像攻击"""
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
        elif attack_method == 'CW':
            attacker = CW(model, 
                          c=self.cw_params['c'].value(),
                          kappa=self.cw_params['kappa'].value(),
                          steps=self.cw_params['steps'].value(),
                          lr=self.cw_params['lr'].value())
        elif attack_method == 'JSMA':
            attacker = JSMA(model, 
                            theta=self.jsma_params['theta'].value(),
                            gamma=self.jsma_params['gamma'].value(),
                            max_iter=self.jsma_params['max_iter'].value(),
                            device=device)
        else: # Should not happen if combo box is synced with attack_methods
            QMessageBox.warning(self, "错误", f"未知的攻击方法: {attack_method}")
            return
            
        # 执行攻击
        try:
            # 临时设置模型为训练模式以启用梯度计算
            model.train()
            x_adv, perturbation = attacker.generate(x, y)
            model.eval()
            
            # 获取预测结果
            with torch.no_grad():
                original_output = model(x)
                adversarial_output = model(x_adv)
                
            original_pred = original_output.argmax(1).item()
            adversarial_pred = adversarial_output.argmax(1).item()
            
            # 可视化结果
            self.single_figure.clear()
            axes = self.single_figure.subplots(1, 3)
            
            # 准备图像数据 - 将32x32的结果放大到128x128用于显示
            from utils.data_loader import denormalize_cifar10
            from PIL import Image
            
            # 反归一化32x32图像
            orig_img_32 = denormalize_cifar10(x.cpu()).squeeze().permute(1, 2, 0).numpy()
            adv_img_32 = denormalize_cifar10(x_adv.cpu()).squeeze().permute(1, 2, 0).numpy()
            pert_32 = perturbation.cpu().squeeze().permute(1, 2, 0).numpy()
            
            # 将32x32图像放大到128x128用于显示
            # orig_img_pil = Image.fromarray((orig_img_32 * 255).astype(np.uint8))
            # orig_img_display = np.array(orig_img_pil.resize((128, 128), Image.NEAREST))
            
            # adv_img_pil = Image.fromarray((adv_img_32 * 255).astype(np.uint8))
            # adv_img_display = np.array(adv_img_pil.resize((128, 128), Image.NEAREST))
            
            # 扰动可视化
            if pert_32.max() > pert_32.min():
                pert_display_32 = (pert_32 - pert_32.min()) / (pert_32.max() - pert_32.min())
            else:
                pert_display_32 = pert_32
            # pert_pil = Image.fromarray((pert_display * 255).astype(np.uint8))
            # pert_display_128 = np.array(pert_pil.resize((128, 128), Image.NEAREST))
            
            # 显示图像
            axes[0].imshow(orig_img_32)
            axes[0].set_title(f'原始图像 (32x32)\n预测: {self.classes[original_pred]}', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(pert_display_32)
            axes[1].set_title('扰动 (32x32)', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(adv_img_32)
            axes[2].set_title(f'对抗样本 (32x32)\n预测: {self.classes[adversarial_pred]}', fontsize=12)
            axes[2].axis('off')
            
            self.single_figure.tight_layout()
            self.single_canvas.draw()
            
            # 更新结果信息
            success = original_pred != adversarial_pred
            confidence_orig = torch.softmax(original_output, dim=1).max().item()
            confidence_adv = torch.softmax(adversarial_output, dim=1).max().item()
            
            self.result_label.setText(
                f"攻击{'成功' if success else '失败'}！ "
                f"原始预测: {self.classes[original_pred]} (置信度: {confidence_orig:.3f}) | "
                f"对抗预测: {self.classes[adversarial_pred]} (置信度: {confidence_adv:.3f})"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"攻击执行失败: {str(e)}")
            
    def init_batch_plot(self):
        """初始化批量分析图表"""
        self.batch_figure.clear()
        ax = self.batch_figure.add_subplot(111)
        ax.set_title("选择分析类型并开始分析", fontsize=14)
        ax.text(0.5, 0.5, "分析结果将在这里显示", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.batch_figure.tight_layout()
        self.batch_canvas.draw()
        
    def start_batch_analysis(self):
        """开始批量分析"""
        if self.parent.get_model() is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
            
        analysis_type = self.analysis_combo.currentText()
        
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)
        QApplication.processEvents() # 确保进度条立即显示

        try:
            if analysis_type == '攻击成功率对比':
                self.analyze_attack_success_rate()
            elif analysis_type == '扰动强度影响':
                self.analyze_epsilon_effect()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析失败: {str(e)}")
            self.analysis_text.append(f"分析失败: {str(e)}")
        finally:
            self.batch_progress_bar.setVisible(False) # 确保分析结束后隐藏进度条
            QApplication.processEvents() 
            
    def analyze_attack_success_rate(self):
        """分析攻击成功率对比 - 使用动态UI参数"""
        self.analysis_text.clear()
        self.analysis_text.append("正在进行攻击成功率对比分析...")
        QApplication.processEvents()

        model = self.parent.get_model()
        device = self.parent.get_device()
        
        _, test_loader, classes = get_cifar10_loaders(batch_size=1)
        sample_size = self.asr_sample_size_spin.value() # Use ASR specific sample size
        
        attack_methods_to_run = []
        if self.fgsm_check.isChecked():
            attack_methods_to_run.append('FGSM')
        if self.pgd_check.isChecked():
            attack_methods_to_run.append('PGD')
        if self.cw_check.isChecked():
            attack_methods_to_run.append('CW')
        if self.jsma_check.isChecked():
            attack_methods_to_run.append('JSMA')

        if not attack_methods_to_run:
            self.analysis_text.append("请至少选择一种攻击方法进行分析。")
            self.batch_progress_bar.setVisible(False)
            return

        self.batch_progress_bar.setMaximum(len(attack_methods_to_run))
        self.batch_progress_bar.setValue(0)
        current_progress = 0
        
        results_methods = []
        success_rates = []
        # Epsilon for ASR is now per-method if applicable
        
        if 'FGSM' in attack_methods_to_run:
            self.analysis_text.append("测试FGSM攻击 (使用指定参数)...")
            QApplication.processEvents()
            fgsm_attacker = FGSM(model, epsilon=self.asr_fgsm_epsilon.value())
            success_rate = self.test_attack_method(fgsm_attacker, 
                                                 test_loader, sample_size, device)
            results_methods.append('FGSM')
            success_rates.append(success_rate)
            current_progress += 1
            self.batch_progress_bar.setValue(current_progress)
            QApplication.processEvents()
            
        if 'PGD' in attack_methods_to_run:
            self.analysis_text.append("测试PGD攻击 (使用指定参数)...")
            QApplication.processEvents()
            pgd_attacker = PGD(model, 
                               epsilon=self.asr_pgd_epsilon.value(), 
                               alpha=self.asr_pgd_alpha.value(), 
                               num_steps=self.asr_pgd_num_steps.value())
            success_rate = self.test_attack_method(pgd_attacker, 
                                                 test_loader, sample_size, device)
            results_methods.append('PGD')
            success_rates.append(success_rate)
            current_progress += 1
            self.batch_progress_bar.setValue(current_progress)
            QApplication.processEvents()
            
        if 'CW' in attack_methods_to_run:
            self.analysis_text.append("测试CW攻击 (使用指定参数)...")
            QApplication.processEvents()
            cw_attacker = CW(model, 
                             c=self.asr_cw_c.value(), 
                             kappa=self.cw_params['kappa'].value(),
                             steps=self.asr_cw_max_iterations.value(),
                             lr=self.cw_params['lr'].value())
            success_rate = self.test_attack_method(cw_attacker, 
                                                 test_loader, sample_size, device)
            results_methods.append('CW')
            success_rates.append(success_rate)
            current_progress += 1
            self.batch_progress_bar.setValue(current_progress)
            QApplication.processEvents()

        if 'JSMA' in attack_methods_to_run:
            self.analysis_text.append("测试JSMA攻击 (使用指定参数)...")
            QApplication.processEvents()
            
            jsma_attacker = JSMA(model, 
                                 theta=self.asr_jsma_theta.value(),
                                 gamma=self.asr_jsma_gamma.value(),
                                 max_iter=self.asr_jsma_max_iterations.value(),
                                 device=device)
            success_rate = self.test_attack_method(jsma_attacker, 
                                                 test_loader, sample_size, device)
            results_methods.append('JSMA')
            success_rates.append(success_rate)
            current_progress += 1
            self.batch_progress_bar.setValue(current_progress)
            QApplication.processEvents()

        self.batch_figure.clear()
        ax = self.batch_figure.add_subplot(111)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        bars = ax.bar(results_methods, success_rates, color=colors[:len(results_methods)])
        
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
                    
        ax.set_ylabel('攻击成功率 (%)', fontsize=12)
        ax.set_title('不同攻击方法的成功率比较', fontsize=14)
        ax.set_ylim(0, max(success_rates) * 1.2 if success_rates else 100)
        ax.grid(axis='y', alpha=0.3)
        
        self.batch_figure.tight_layout()
        self.batch_canvas.draw()
        
        self.analysis_text.append("\n分析完成！结果摘要:")
        for method, rate in zip(results_methods, success_rates):
            self.analysis_text.append(f"{method}: {rate:.1f}%")
        QApplication.processEvents()
        
    def analyze_epsilon_effect(self):
        """分析扰动强度影响"""
        self.analysis_text.clear()
        self.analysis_text.append("正在分析扰动强度对攻击成功率的影响...")
        QApplication.processEvents()

        model = self.parent.get_model()
        device = self.parent.get_device()
        
        _, test_loader, classes = get_cifar10_loaders(batch_size=1)
        # For epsilon effect, sample_size comes from the epsilon_effect_params_group
        sample_size = min(50, self.sample_size_spin.value()) 
        
        epsilon_min_val = self.epsilon_min.value()
        epsilon_max_val = self.epsilon_max.value()
        epsilon_steps_count = self.epsilon_steps.value()
        epsilons = np.linspace(epsilon_min_val, epsilon_max_val, epsilon_steps_count)
        
        num_methods_checked = 0
        if self.fgsm_check.isChecked() and self.fgsm_check.isEnabled(): # Check if enabled
            num_methods_checked +=1
        if self.pgd_check.isChecked() and self.pgd_check.isEnabled():   # Check if enabled
            num_methods_checked +=1

        if num_methods_checked == 0:
            self.analysis_text.append("请至少选择FGSM或PGD进行ε影响分析。")
            self.batch_progress_bar.setVisible(False)
            return

        total_progress_steps = epsilon_steps_count * num_methods_checked
        self.batch_progress_bar.setMaximum(total_progress_steps)
        self.batch_progress_bar.setValue(0)
        current_progress = 0

        self.batch_figure.clear()
        ax = self.batch_figure.add_subplot(111)
        
        if self.fgsm_check.isChecked():
            fgsm_rates = []
            for eps in epsilons:
                self.analysis_text.append(f"测试FGSM (ε={eps:.3f})...")
                QApplication.processEvents()
                attacker = FGSM(model, epsilon=eps)
                rate = self.test_attack_method(attacker, test_loader, sample_size, device)
                fgsm_rates.append(rate)
                current_progress += 1
                self.batch_progress_bar.setValue(current_progress)
                QApplication.processEvents()
            ax.plot(epsilons, fgsm_rates, 'o-', label='FGSM', linewidth=2, markersize=6)
            
        if self.pgd_check.isChecked():
            pgd_rates = []
            for eps in epsilons:
                self.analysis_text.append(f"测试PGD (ε={eps:.3f})...")
                QApplication.processEvents()
                attacker = PGD(model, epsilon=eps, num_steps=10) # Assuming PGD defaults here
                rate = self.test_attack_method(attacker, test_loader, sample_size, device)
                pgd_rates.append(rate)
                current_progress += 1
                self.batch_progress_bar.setValue(current_progress)
                QApplication.processEvents()
            ax.plot(epsilons, pgd_rates, 's-', label='PGD', linewidth=2, markersize=6)
            
        ax.set_xlabel('扰动强度 (ε)', fontsize=12)
        ax.set_ylabel('攻击成功率 (%)', fontsize=12)
        ax.set_title('扰动强度对攻击成功率的影响', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        self.batch_figure.tight_layout()
        self.batch_canvas.draw()
        
        self.analysis_text.append("\n扰动强度影响分析完成！")
        QApplication.processEvents()

    def test_attack_method(self, attacker, test_loader, sample_size, device):
        """测试攻击方法"""
        total_samples = 0
        successful_attacks = 0
        
        model = self.parent.get_model()
        
        for i, (x, y) in enumerate(test_loader):
            if total_samples >= sample_size:
                break
                
            x, y = x.to(device), y.to(device)
            
            # 获取原始预测
            with torch.no_grad():
                original_output = model(x)
                original_pred = original_output.argmax(1)
            
            # 只对正确分类的样本进行攻击
            if original_pred.item() == y.item():
                try:
                    # 临时设置为训练模式用于攻击
                    model.train()
                    
                    # 生成对抗样本
                    x_adv, _ = attacker.generate(x, y)
                    
                    # 恢复为评估模式
                    model.eval()
                    
                    # 获取对抗预测
                    with torch.no_grad():
                        adv_output = model(x_adv)
                        adv_pred = adv_output.argmax(1)
                    
                    # 检查攻击是否成功
                    if adv_pred.item() != original_pred.item():
                        successful_attacks += 1
                        
                    total_samples += 1
                    
                except Exception as e:
                    # 确保无论如何都恢复为评估模式
                    model.eval()
                    print(f"攻击失败：{e}")
                    continue
        
        # 确保最后恢复为评估模式
        model.eval()
                    
        if total_samples == 0:
            return 0.0
            
        success_rate = (successful_attacks / total_samples) * 100
        return success_rate