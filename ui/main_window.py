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
        
        # 创建菜单栏
        self.create_menubar()
        
        # 创建标签页
        self.create_tabs()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f'设备: {self.device} | 模型未加载')
        
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
    
    def create_menubar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        # 加载模型
        load_model_action = QAction('加载模型', self)
        load_model_action.setShortcut('Ctrl+O')
        load_model_action.triggered.connect(self.load_model)
        file_menu.addAction(load_model_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        

    def create_tabs(self):
        """创建标签页"""
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # 攻击模块标签页
        self.attack_widget = AttackWidget(self)
        self.tab_widget.addTab(self.attack_widget, "攻击与分析")
        
        # 防御模块标签页
        self.defense_widget = DefenseWidget(self)
        self.tab_widget.addTab(self.defense_widget, "防御模块")
    
    def load_model(self):
        """加载模型"""
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pth);;所有文件 (*.*)"
        )
        
        if model_path:
            try:
                self.model = SimpleCNN().to(self.device)
                
                # 尝试加载模型
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 检查是否是完整的checkpoint还是只有state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict):
                    # 假设这是一个state_dict
                    self.model.load_state_dict(checkpoint)
                else:
                    # 可能是旧格式的模型
                    self.model.load_state_dict(checkpoint)
                    
                self.model.eval()
                
                model_name = model_path.split("/")[-1]
                self.status_bar.showMessage(f'设备: {self.device} | 已加载模型: {model_name}')
                
                # 通知各个标签页模型已加载
                self.attack_widget.on_model_loaded()
                self.defense_widget.on_model_loaded()
                
                QMessageBox.information(self, "成功", f"模型 {model_name} 加载成功！")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败：{str(e)}")

    
    def get_model(self):
        """获取当前模型"""
        return self.model
    
    def get_device(self):
        """获取当前设备"""
        return self.device 