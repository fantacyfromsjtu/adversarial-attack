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
from defenses import AdversarialTraining
from models import SimpleCNN
from utils import get_cifar10_loaders
from config import DEFENSE_PARAMS, MODEL_DIR
import os

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
        
        # 防御方法显示 (替换下拉列表)
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("<b>防御方法: 对抗训练</b>")) # 直接显示方法
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
        # 参数设置
        self.param_group = QGroupBox("对抗训练参数")
        param_layout = QFormLayout(self.param_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(DEFENSE_PARAMS.get('adversarial_training', {}).get('epochs', 10)) # Safely get params
        param_layout.addRow("训练轮数 (Epochs):", self.epochs_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(DEFENSE_PARAMS.get('adversarial_training', {}).get('learning_rate', 0.001)) # Safely get params
        self.lr_spin.setDecimals(4)
        param_layout.addRow("学习率 (Learning Rate):", self.lr_spin)
        
        # Adversarial Training specific
        self.epsilon_label = QLabel("扰动强度 (ε):")
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1.0)
        self.epsilon_spin.setSingleStep(0.01)
        self.epsilon_spin.setValue(DEFENSE_PARAMS.get('adversarial_training', {}).get('epsilon', 0.03)) # Safely get params
        param_layout.addRow(self.epsilon_label, self.epsilon_spin)
        
        layout.addWidget(self.param_group)
        
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
        self.evaluate_btn.setEnabled(False) # 评估按钮初始不可用，直到有模型为止
        button_layout.addWidget(self.evaluate_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 训练历史图表
        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMaximumHeight(300) # 限制图表最大高度
        layout.addWidget(self.canvas)
        
        # 信息显示
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150) # 限制信息框最大高度
        self.info_text.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.info_text)
        
        self.init_plot()
        
    def init_plot(self):
        """初始化图表"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title("训练历史", fontsize=14)
        ax.text(0.5, 0.5, "开始训练后将显示训练历史", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off') # Initially turn off axis if no data
        self.figure.tight_layout()
        self.canvas.draw()
        
    def on_model_loaded(self):
        """模型加载完成的处理，由主窗口调用"""
        self.evaluate_btn.setEnabled(True)
        self.train_btn.setEnabled(True) # 模型加载后可以开始防御训练
        self.info_text.append("模型已加载，可以开始对抗训练或评估。")
        
    def start_training(self):
        """开始训练"""
        if not (hasattr(self.parent, 'model') and self.parent.model):
            QMessageBox.warning(self, "警告", "请先从主窗口加载一个模型以进行对抗训练！")
            return

        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "警告", "训练正在进行中！")
            return
            
        # 获取参数 (仅对抗训练)
        epochs = self.epochs_spin.value()
        learning_rate = self.lr_spin.value()
        epsilon = self.epsilon_spin.value()
            
        # 创建训练线程
        self.training_thread = TrainingThread(
            epochs=epochs, 
            learning_rate=learning_rate, 
            epsilon=epsilon,
            parent_widget=self # Pass self to access student model via self.parent.model
        )
        self.training_thread.progress_updated.connect(self.update_progress)
        self.training_thread.epoch_completed.connect(self.update_plot_from_history) # Changed slot name for clarity
        self.training_thread.training_completed.connect(self.on_training_finished) # Changed slot name
        self.training_thread.info_updated.connect(self.update_info)
        
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.evaluate_btn.setEnabled(False) # 训练时不可评估
        self.progress_bar.setValue(0)
        self.info_text.clear()
        self.info_text.append("开始对抗训练...")
        
        self.training_thread.start()
            
    def stop_training(self):
        """停止训练"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            # on_training_finished will be called by the thread's signal
        else:
            self.info_text.append("没有正在进行的训练。")
            
    def on_training_finished(self, success, message, results): # Updated to match signal
        """训练完成或停止后的处理"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.evaluate_btn.setEnabled(True) # 训练结束后可以评估
        self.info_text.append(message)
        if success and results:
            self.info_text.append(f"训练后的模型保存在: {results.get('model_path', 'N/A')}")
            if 'history' in results:
                 self.update_plot_from_history(results['history']) # Plot final history
        # If stopped early, results might contain partial history
        elif not success and results and 'history' in results and results['history']['train_loss']: # Check if any history exists
            self.update_plot_from_history(results['history'])

        if self.training_thread:
            self.training_thread.quit() # Ensure thread resources are cleaned up
            self.training_thread.wait()
            self.training_thread = None
            
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def update_info(self, text):
        """更新信息显示"""
        self.info_text.append(text)
        QApplication.processEvents() # Ensure GUI updates
        
    def update_plot_from_history(self, history_dict): # Renamed and updated
        """根据历史记录更新训练历史图表"""
        self.figure.clear()
        
        epochs_ran = len(history_dict.get('train_loss', []))
        if epochs_ran == 0:
            self.init_plot() # Reset to initial "no data" state
            return

        epochs_range = range(1, epochs_ran + 1)
        
        ax1 = self.figure.add_subplot(121)
        ax1.plot(epochs_range, history_dict['train_loss'], 'b-o', markersize=4, label='训练损失')
        if 'val_loss' in history_dict and len(history_dict['val_loss']) == epochs_ran:
            ax1.plot(epochs_range, history_dict['val_loss'], 'r-s', markersize=4, label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('损失曲线')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2 = self.figure.add_subplot(122)
        ax2.plot(epochs_range, history_dict['train_acc'], 'b-o', markersize=4, label='训练准确率')
        if 'val_acc' in history_dict and len(history_dict['val_acc']) == epochs_ran:
            ax2.plot(epochs_range, history_dict['val_acc'], 'r-s', markersize=4, label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('准确率曲线')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def evaluate_model(self):
        """评估当前通过防御训练得到的模型（或主窗口加载的模型）"""
        if not (hasattr(self.parent, 'model') and self.parent.model):
            QMessageBox.warning(self, "警告", "请先加载或训练一个模型！")
            return

        model_to_evaluate = self.parent.model # Evaluate the model managed by MainWindow
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_to_evaluate.to(device)
        model_to_evaluate.eval()

        _, test_loader, _ = get_cifar10_loaders()
        
        self.info_text.append("\n开始评估当前模型...")

        # 评估干净样本
        clean_correct = 0
        clean_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model_to_evaluate(data)
                _, predicted = torch.max(outputs.data, 1)
                clean_total += target.size(0)
                clean_correct += (predicted == target).sum().item()
        clean_accuracy = 100 * clean_correct / clean_total if clean_total > 0 else 0
        self.info_text.append(f"干净样本准确率: {clean_accuracy:.2f}%")

        # 评估对抗样本 (使用默认FGSM攻击进行评估)
        # 可以考虑让用户选择评估时使用的攻击方法和参数
        eval_attacker = FGSM(model_to_evaluate, epsilon=self.epsilon_spin.value()) # Use current epsilon for eval
        adv_correct = 0
        adv_total = 0
        
        # Important: model might need to be in train() mode for some attacks if they use batchnorm etc.
        # For FGSM, eval() is fine.
        # original_mode = model_to_evaluate.training
        # model_to_evaluate.train() # Or eval() if attack allows

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            adv_data, _ = eval_attacker.generate(data, target) # FGSM's generate handles model mode internally if needed
            
            with torch.no_grad(): # Ensure no gradients for output calculation
                outputs = model_to_evaluate(adv_data)
                _, predicted = torch.max(outputs.data, 1)
            adv_total += target.size(0)
            adv_correct += (predicted == target).sum().item()
        
        # model_to_evaluate.train(original_mode) # Restore original mode

        adv_accuracy = 100 * adv_correct / adv_total if adv_total > 0 else 0
        self.info_text.append(f"FGSM 对抗样本准确率 (ε={eval_attacker.epsilon:.3f}): {adv_accuracy:.2f}%")
        self.info_text.append("评估完成。")
        QApplication.processEvents()


class TrainingThread(QThread):
    """训练线程"""
    progress_updated = pyqtSignal(int)
    epoch_completed = pyqtSignal(dict) 
    training_completed = pyqtSignal(bool, str, dict) 
    info_updated = pyqtSignal(str)

    def __init__(self, epochs, learning_rate, epsilon, parent_widget): # 移除了蒸馏参数
        super().__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon # 仅对抗训练需要
        self.parent_widget = parent_widget

        self._is_running = True
        self.model = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def stop(self):
        """停止线程"""
        self._is_running = False
        self.info_updated.emit("训练停止指令已发送...")

    def run(self):
        """执行训练任务 (仅对抗训练)"""
        self._is_running = True
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        try:
            self.info_updated.emit(f"初始化对抗训练...")

            if not (hasattr(self.parent_widget.parent, 'model') and self.parent_widget.parent.model):
                self.info_updated.emit("错误: 执行对抗训练前，请先在主窗口加载一个模型。")
                self.training_completed.emit(False, "对抗训练失败: 学生模型未加载。", {})
                return

            self.model = self.parent_widget.parent.model
            self.model.to(self.device)

            train_loader, test_loader, _ = get_cifar10_loaders(batch_size=128) 

            self.info_updated.emit(f"使用对抗训练，扰动强度 ε={self.epsilon}")
            trainer_instance = AdversarialTraining(self.model, self.device, epsilon=self.epsilon)
            trainer_instance.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            for epoch in range(1, self.epochs + 1):
                if not self._is_running:
                    self.info_updated.emit("训练已手动停止。")
                    break

                self.info_updated.emit(f"Epoch {epoch}/{self.epochs} 开始...")
                self.model.train() # 确保模型在训练模式
                train_loss, train_acc = trainer_instance.train_epoch(train_loader, epoch)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)

                val_acc, val_loss = self.validate(test_loader, self.device, 
                                                  use_adv_eval=True, # 对抗训练在验证时也使用对抗样本
                                                  trainer_instance=trainer_instance)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                self.epoch_completed.emit({'epoch': epoch, 
                                           'train_loss': train_loss, 'train_acc': train_acc,
                                           'val_loss': val_loss, 'val_acc': val_acc,
                                           **history # Send full history for live plotting if needed
                                           })
                self.progress_updated.emit(int((epoch / self.epochs) * 100))
                self.info_updated.emit(f"Epoch {epoch}/{self.epochs} 完成: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}%, 验证损失={val_loss:.4f}, 验证准确率={val_acc:.2f}%")

            if self._is_running:
                self.info_updated.emit("对抗训练成功完成。")
                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)

                base_model_name = "model" 
                current_model_name_in_main_window = "unknown_model"
                if hasattr(self.parent_widget.parent, 'status_bar') and self.parent_widget.parent.status_bar.currentMessage().startswith("设备:") and "已加载模型: " in self.parent_widget.parent.status_bar.currentMessage():
                    try:
                        msg = self.parent_widget.parent.status_bar.currentMessage()
                        current_model_name_in_main_window = msg.split("已加载模型: ")[-1]
                    except:
                        pass # keep default
                base_model_name = os.path.splitext(current_model_name_in_main_window)[0]

                saved_model_name = f"{base_model_name}_adv_trained.pth"
                saved_model_path = os.path.join(MODEL_DIR, saved_model_name)

                torch.save({'model_state_dict': self.model.state_dict(), 
                            'epoch': self.epochs,
                            'model_name': saved_model_name}, 
                           saved_model_path)
                self.info_updated.emit(f"对抗训练后的模型已保存到: {saved_model_path}")
                # Update main window's current model to this newly trained one
                if hasattr(self.parent_widget.parent, 'load_model_from_path'):
                    self.parent_widget.parent.load_model_from_path(saved_model_path, update_status=False) 
                    self.info_updated.emit(f"主窗口模型已更新为: {saved_model_name}")

                self.training_completed.emit(True, "对抗训练成功完成。", {'model_path': saved_model_path, 'history': history})
            else: # Training was stopped
                self.training_completed.emit(False, "对抗训练已停止。", {'history': history})

        except Exception as e:
            self.info_updated.emit(f"对抗训练过程中发生错误: {e}")
            import traceback
            self.info_updated.emit(traceback.format_exc())
            self.training_completed.emit(False, f"对抗训练失败: {e}", {'history': history})


    def validate(self, test_loader, device, use_adv_eval=False, trainer_instance=None):
        """验证模型"""
        if not self.model:
            self.info_updated.emit("错误: 验证时模型未初始化。")
            return 0, float("inf")

        self.model.eval()  # 设置为评估模式
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        for data, target in test_loader:
            data_eval, target_eval = data.to(device), target.to(device)

            if use_adv_eval and trainer_instance and hasattr(trainer_instance, "attacker"):
                # 对抗训练的验证集也使用对抗样本
                # 为生成对抗样本启用梯度计算
                data_eval = data_eval.clone().detach().requires_grad_(True)

                # 确保攻击器使用当前评估模式的模型
                original_attacker_model_mode = trainer_instance.attacker.model.training
                trainer_instance.attacker.model.eval()

                # 生成对抗样本 - 需要梯度计算，所以不能在 torch.no_grad() 中
                adv_data, _ = trainer_instance.attacker.generate(data_eval, target_eval)

                trainer_instance.attacker.model.train(
                    original_attacker_model_mode
                )  # 恢复攻击器模型的原始模式
                data_eval = adv_data.detach()  # 分离梯度，避免后续计算中的梯度传播

            # 模型预测和损失计算
            with torch.no_grad():
                outputs = self.model(data_eval)
                loss = criterion(outputs, target_eval)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target_eval.size(0)
                correct += predicted.eq(target_eval).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float("inf")

        # self.model.train() # 在run循环的下一轮迭代开始时会设置 train()
        return accuracy, avg_loss
