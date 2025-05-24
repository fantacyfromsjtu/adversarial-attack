"""
主窗口UI
"""
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
from models import SimpleCNN
from config import WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT
from .attack_widget import AttackWidget
from .defense_widget import DefenseWidget
from .visualization_widget import VisualizationWidget


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initUI()
    
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # 设置中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建工具栏
        self.create_toolbar()
        
        # 创建标签页
        self.create_tabs()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f'设备: {self.device}')
        
        # 应用样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4CAF50;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = self.addToolBar('工具栏')
        toolbar.setMovable(False)
        
        # 加载模型按钮
        load_model_action = QAction(QIcon(), '加载模型', self)
        load_model_action.triggered.connect(self.load_model)
        toolbar.addAction(load_model_action)
        
        # 训练模型按钮
        train_model_action = QAction(QIcon(), '训练模型', self)
        train_model_action.triggered.connect(self.train_model)
        toolbar.addAction(train_model_action)
        
        toolbar.addSeparator()
        
        # 模型信息标签
        self.model_info_label = QLabel('模型未加载')
        toolbar.addWidget(self.model_info_label)
    
    def create_tabs(self):
        """创建标签页"""
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # 攻击模块标签页
        self.attack_widget = AttackWidget(self)
        self.tab_widget.addTab(self.attack_widget, "攻击模块")
        
        # 防御模块标签页
        self.defense_widget = DefenseWidget(self)
        self.tab_widget.addTab(self.defense_widget, "防御模块")
        
        # 可视化模块标签页
        self.visualization_widget = VisualizationWidget(self)
        self.tab_widget.addTab(self.visualization_widget, "可视化")
    
    def load_model(self):
        """加载模型"""
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pth);;所有文件 (*.*)"
        )
        
        if model_path:
            try:
                self.model = SimpleCNN().to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                self.model_info_label.setText(f'已加载模型: {model_path.split("/")[-1]}')
                self.status_bar.showMessage('模型加载成功！')
                
                # 通知各个标签页模型已加载
                self.attack_widget.on_model_loaded()
                self.defense_widget.on_model_loaded()
                self.visualization_widget.on_model_loaded()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败：{str(e)}")
    
    def train_model(self):
        """训练模型"""
        reply = QMessageBox.question(
            self, '训练模型', 
            '是否要训练一个新的CNN模型？\n这将需要一些时间。',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 这里可以打开一个训练对话框或启动训练线程
            QMessageBox.information(self, "提示", "训练功能将在后续版本中实现。")
    
    def get_model(self):
        """获取当前模型"""
        return self.model
    
    def get_device(self):
        """获取当前设备"""
        return self.device 