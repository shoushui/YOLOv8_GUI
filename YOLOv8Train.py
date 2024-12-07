'''
@Project ：PyGUI 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：Dqs
@Date    ：2024/10/19 19:41 
'''
import shutil
import sys
import os

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import torch
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from YOLOTrain import Ui_MainWindow
from ultralytics import YOLO
from ultralytics.utils import LOGGER

import os
PROJECT_DIR = os.path.dirname(__file__)
current_directory = os.getcwd()
# 自定义日志流处理器
class LogStreamHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        log_entry = self.format(record)
        if any(char.isalpha() for char in log_entry):  # 只保留包含字母的日志信息
            self.signal.emit(log_entry)

# 创建一个独立的 Logger
def get_thread_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    return logger

# 创建一个用于训练的线程
class TrainThread(QThread):
    # 定义信号，用于在线程中传递消息到主线程
    signal_train_start = pyqtSignal(str)
    signal_train_done = pyqtSignal(str)
    signal_train_error = pyqtSignal(str)
    signal_train_log = pyqtSignal(str)  # 新增日志信号

    def __init__(self, model_yaml_path, model_weight_path, data_yaml_path, epoches, imgsz, batch_size, workers,
                 early_stop, optimizer, device):
        super(TrainThread, self).__init__()
        self.model_yaml_path = model_yaml_path
        self.model_weight_path = model_weight_path
        self.data_yaml_path = data_yaml_path
        self.epoches = epoches
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.workers = workers
        self.early_stop = early_stop
        self.optimizer = optimizer
        self.device = device

        # 创建训练独立的 Logger
        self.train_logger = get_thread_logger("train_logger")

    def run(self):
        try:
            # 发出训练开始信号
            self.signal_train_start.emit("训练开始...")
            self.signal_train_log.emit("加载模型...")

            # 加载模型并进行训练
            model = YOLO(self.model_yaml_path)
            model.load(self.model_weight_path)
            self.signal_train_log.emit("模型加载成功，开始训练...")


            # 创建自定义日志处理器
            log_handler = LogStreamHandler(self.signal_train_log)
            self.train_logger.addHandler(log_handler)
            logging.getLogger().setLevel(logging.INFO)  # 设置日志级别

            # 连接 LOGGER 的处理器到 log_handler
            LOGGER.addHandler(log_handler)


            model.train(
                data=self.data_yaml_path,
                epochs=int(self.epoches),
                imgsz=int(self.imgsz),
                batch=int(self.batch_size),
                workers=int(self.workers),
                multi_scale=True,
                cache=False,
                device=self.device,
                amp=False,
                augment=True,
                cos_lr=True,
                optimizer=self.optimizer,
                patience=int(self.early_stop),
                resume=False
            )

            # 发出训练完成信号
            self.signal_train_done.emit("训练完成！")

        except Exception as e:
            self.signal_train_error.emit(f"训练失败: {str(e)}")
            self.signal_train_log.emit(f"训练失败: {str(e)}")
        finally:
            # 移除自定义日志处理器
            self.train_logger.removeHandler(log_handler)
            LOGGER.removeHandler(log_handler)  # 移除 LOGGER 的处理器
            self.exit()  # 退出线程的优雅方法


# 创建一个用于导出的线程
class ExportThread(QThread):
    # 定义信号，用于在线程中传递消息到主线程
    signal_export_start = pyqtSignal(str)
    signal_export_done = pyqtSignal(str)
    signal_export_error = pyqtSignal(str)
    signal_export_log = pyqtSignal(str)  # 新增日志信号

    def __init__(self, onnx_yaml_path, onnx_path, onnx_opset_version, onnx_input_size):
        super(ExportThread, self).__init__()
        self.onnx_yaml_path = onnx_yaml_path
        self.onnx_path = onnx_path
        self.onnx_opset_version = onnx_opset_version
        self.onnx_input_size = onnx_input_size

        # 创建导出独立的 Logger
        self.export_logger = get_thread_logger("export_logger")

    def run(self):
        try:
            # 发出导出开始信号
            self.signal_export_start.emit("导出开始...")
            self.signal_export_log.emit("加载模型...")

            # 加载模型并进行导出
            model = YOLO(self.onnx_yaml_path)
            model.load(self.onnx_path)
            self.signal_export_log.emit("模型加载成功，开始导出...")

            # 创建自定义日志处理器
            export_log_handler = LogStreamHandler(self.signal_export_log)
            self.export_logger.addHandler(export_log_handler)
            logging.getLogger().setLevel(logging.INFO)  # 设置日志级别

            # 连接 LOGGER 的处理器到 log_handler
            LOGGER.addHandler(export_log_handler)

            # 导出模型
            model.export(format="onnx", opset=int(self.onnx_opset_version), imgsz=int(self.onnx_input_size))

            # 发出导出完成信号
            self.signal_export_done.emit("导出成功，模型已成功导出为 ONNX 格式！")

        except Exception as e:
            self.signal_export_error.emit(f"导出失败: {str(e)}")
            self.signal_export_log.emit(f"导出失败: {str(e)}")

        finally:
            # 移除自定义日志处理器
            self.export_logger.removeHandler(export_log_handler)
            LOGGER.removeHandler(export_log_handler)  # 移除 LOGGER 的处理器
            self.exit()  # 退出线程的优雅方法

class YOLOActivate(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(YOLOActivate, self).__init__()
        self.setupUi(self)
        # 训练参数
        self.epoches = self.spinbox_train_epoches.text()
        self.workers = self.spinbox_train_workers.text()
        self.batch_size = self.spinbox_train_batch_size.text()
        self.imgsz = self.spinbox_train_imgsz.text()
        self.early_stop = self.spinbox_train_earlystop.text()

        self.optimizer = self.train_optimizer()
        self.device = self.train_device()
        self.model_yaml_path = None
        self.data_yaml_path = None
        self.model_weight_path = None
        self.show_image_path = os.path.join(PROJECT_DIR,"save_model/Loss_PR_MAP.png")

        # 导出参数
        self.onnx_path = None
        self.onnx_yaml_path = None
        self.onnx_opset_version = self.comboBox_export_onnx_opset.currentText()
        self.onnx_input_size = self.spinbox_export_onnx_inputsize.text()

        # 训练槽函数
        self.button_model_yaml.clicked.connect(self.train_model_yaml_pushButton)
        self.button_data_yaml.clicked.connect(self.train_data_yaml_pushButton)
        self.button_pretrain_model.clicked.connect(self.train_pretrain_model_pushButton)

        self.spinbox_train_epoches.valueChanged.connect(self.train_epochs_spinbox)
        self.spinbox_train_workers.valueChanged.connect(self.train_workers_spinbox)
        self.spinbox_train_batch_size.valueChanged.connect(self.train_batchsize_spinbox)
        self.spinbox_train_imgsz.valueChanged.connect(self.train_imgsize_spinbox)
        self.spinbox_train_earlystop.valueChanged.connect(self.train_earlystop_spinbox)

        self.button_train_checkparams.clicked.connect(self.train_CheckButton)
        self.button_train_begin.setEnabled(False)  # 禁用训练按钮
        self.button_train_begin.clicked.connect(self.train_pushButton)



        # 导出槽函数
        self.button_export_model_yaml.clicked.connect(self.export_model_yaml_pushButton)
        self.button_trained_model.clicked.connect(self.export_model_pushButton)
        self.spinbox_export_onnx_inputsize.valueChanged.connect(self.export_model_size_spinbox)
        self.comboBox_export_onnx_opset.currentIndexChanged.connect(self.export_model_opset_comboBox)

        self.button_export_checkparams.clicked.connect(self.export_CheckButton)
        self.button_export_begin.setEnabled(False)  # 禁用训练按钮
        self.button_export_begin.clicked.connect(self.export_pushButton)

    def train_model_yaml_pushButton(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "select", "", "文本文件 (*.yaml)")
        if file_name:
            self.line_model_yaml_info.setText(file_name)
            # print(f"模型配置文件的地址为：{self.line_model_yaml_info.text()}")
            self.model_yaml_path = self.line_model_yaml_info.text()
    def train_data_yaml_pushButton(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "select", "", "文本文件 (*.yaml)")
        if file_name:
            self.line_data_yaml_info.setText(file_name)
            # print(f"数据配置文件的地址为：{self.line_data_yaml_info.text()}")
            self.data_yaml_path = self.line_data_yaml_info.text()
    def train_pretrain_model_pushButton(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "select", "", "文本文件 (*.pt)")
        if file_name:
            self.line_pretrain_model_info.setText(file_name)
            # print(f"预训练模型文件的地址为：{self.line_pretrain_model_info.text()}")
            self.model_weight_path = self.line_pretrain_model_info.text()
    def train_epochs_spinbox(self, value):
        self.epoches = value
        # print(f"训练轮次为：{self.epoches}")
    def train_workers_spinbox(self, value):
        self.workers = value
        # print(f"工作线程数量为：{self.workers}")
    def train_imgsize_spinbox(self, value):
        self.imgsz = value
        # print(f"训练图形大小为：{self.imgsz}")
    def train_batchsize_spinbox(self, value):
        self.batch_size = value
        # print(f"训练批次为：{self.batch_size}")
    def train_earlystop_spinbox(self, value):
        self.early_stop = value
        # print(f"训练早停为：{self.batch_size}")
    def train_optimizer(self):
        if self.checkbox_train_adam.isChecked():
            optimizer = self.checkbox_train_adam.text()
        else:
            optimizer = self.checkbox_train_sgd.text()
        return optimizer
    def train_device(self):
        if self.checkbox_gpu.isChecked():
            device = torch.cuda.current_device()
        else:
            device = "cpu"

        return device

    def update_image(self, log_message):
        if os.path.exists(self.show_image_path) and ("Epoch" in log_message or "all" in log_message):
            pixmap = QPixmap(self.show_image_path)
            if pixmap.isNull():
                logging.info("Failed to load image. QPixmap is null.")  # 调试信息
            else:
                self.label_train_accuracy.setPixmap(pixmap.scaled(self.label_train_accuracy.size(), Qt.KeepAspectRatio))
                self.label_train_accuracy.setScaledContents(True)  # 自适应 QLabel 的大小


    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message, QMessageBox.Ok)
    def train_CheckButton(self):
        if self.model_yaml_path and self.data_yaml_path and self.model_weight_path:
            flag = True

            if os.path.getsize(self.model_yaml_path) == 0:  # 如果文件大小为0:
                flag = False
                self.show_warning("警告",f"{self.model_yaml_path}内容为空，请重新选择文件！")
                self.button_train_begin.setEnabled(False)  # 禁用训练按钮

            if os.path.getsize(self.data_yaml_path) == 0:
                flag = False
                self.show_warning("警告",f"{self.data_yaml_path}内容为空，请重新选择文件！")
                self.button_train_begin.setEnabled(False)  # 禁用训练按钮

            if os.path.getsize(self.model_weight_path) == 0:  # 如果文件大小为0
                flag = False
                self.show_warning("警告", "选择的文件大小为0，请重新选择文件!")
                self.button_train_begin.setEnabled(False)  # 禁用训练按钮
            if flag:
                self.show_warning("参数检查", "参数检查成功！")  # 弹出警告框
                self.button_train_begin.setEnabled(True)  # 禁用训练按钮
        else:
            self.show_warning("参数检查", "参数检查失败！前三项参数有未选参数！！！")  # 弹出警告框
            self.button_train_begin.setEnabled(False)  # 禁用训练按钮
    def train_pushButton(self):
        # 创建并启动训练线程
        self.train_thread = TrainThread(
            model_yaml_path=self.model_yaml_path,
            model_weight_path=self.model_weight_path,
            data_yaml_path=self.data_yaml_path,
            epoches=int(self.epoches),
            imgsz=int(self.imgsz),
            batch_size=int(self.batch_size),
            workers=int(self.workers),
            early_stop=int(self.early_stop),
            optimizer=self.optimizer,
            device=self.device
        )

        # 连接线程的信号与主界面更新槽函数
        self.train_thread.signal_train_start.connect(self.on_train_start)
        self.train_thread.signal_train_done.connect(self.on_train_done)
        self.train_thread.signal_train_error.connect(self.on_train_error)
        self.train_thread.signal_train_log.connect(self.on_train_log)
        self.train_thread.signal_train_log.connect(self.update_image)

        # 启动线程
        self.train_thread.start()

    def on_train_start(self, message):
        self.button_train_begin.setEnabled(False)
        QMessageBox.information(self, "训练信息", message)

    def on_train_done(self, message):
        self.button_train_begin.setEnabled(True)
        QMessageBox.information(self, "训练信息", message)
        shutil.rmtree(os.path.join(PROJECT_DIR,'runs'), ignore_errors=True)
    def on_train_error(self, message):
        self.button_train_begin.setEnabled(True)
        QMessageBox.warning(self, "训练错误", message)
        torch.cuda.empty_cache()  # 清除 GPU 缓存
    def on_train_log(self, log_message):
        # 将日志信息添加到QPlainTextEdit中
        self.line_train_logs.appendPlainText(log_message)

    def export_model_pushButton(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "select", "", "文本文件 (*.pt)")
        if file_name:
            self.line_trained_model_info.setText(file_name)
            self.onnx_path = self.line_trained_model_info.text()

    def export_model_opset_comboBox(self):
        self.onnx_opset_version = self.comboBox_export_onnx_opset.currentText()

    def export_model_yaml_pushButton(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "select", "", "文本文件 (*.yaml)")
        if file_name:
            self.line_export_model_yaml_info.setText(file_name)
            self.onnx_yaml_path = self.line_export_model_yaml_info.text()

    def export_model_size_spinbox(self, value):
        self.onnx_input_size = value

    def export_CheckButton(self):
        if self.onnx_path and self.onnx_yaml_path and self.onnx_input_size:
            flag = True

            if os.path.getsize(self.onnx_yaml_path) == 0:  # 如果文件大小为0:
                flag = False
                self.show_warning("警告",f"{self.onnx_yaml_path}内容为空，请重新选择文件！")
                self.button_export_begin.setEnabled(False)  # 禁用训练按钮

            if os.path.getsize(self.onnx_path) == 0:  # 如果文件大小为0
                flag = False
                self.show_warning("警告", f"选择的{self.onnx_path}文件大小为0，请重新选择文件!")
                self.button_export_begin.setEnabled(False)  # 禁用训练按钮
            if flag:
                self.show_warning("参数检查", "参数检查成功！")  # 弹出警告框
                self.button_export_begin.setEnabled(True)  # 禁用训练按钮
        else:
            self.show_warning("参数检查", "参数检查失败！！")  # 弹出警告框
            self.button_export_begin.setEnabled(False)  # 禁用训练按钮

    def export_pushButton(self):
        # 创建并启动导出线程
        self.export_thread = ExportThread(
            onnx_yaml_path=self.onnx_yaml_path,
            onnx_path=self.onnx_path,
            onnx_opset_version=self.onnx_opset_version,
            onnx_input_size=self.onnx_input_size
        )

        # 连接线程的信号与主界面更新槽函数
        self.export_thread.signal_export_start.connect(self.on_export_start)
        self.export_thread.signal_export_done.connect(self.on_export_done)
        self.export_thread.signal_export_error.connect(self.on_export_error)
        self.export_thread.signal_export_log.connect(self.on_export_log)

        # 启动线程
        self.export_thread.start()

    def on_export_start(self, message):
        self.button_export_begin.setEnabled(False)
        QMessageBox.information(self, "导出信息", message)

    def on_export_done(self, message):
        self.button_export_begin.setEnabled(True)
        QMessageBox.information(self, "导出信息", message)

    def on_export_error(self, message):
        self.button_export_begin.setEnabled(True)
        QMessageBox.warning(self, "导出错误", message)
        torch.cuda.empty_cache()  # 清除 GPU 缓存

    def on_export_log(self, log_message):
        # 将日志信息添加到QLineEdit中
        self.line_export_logs.appendPlainText(log_message)




if __name__ == '__main__':
    app = QApplication(sys.argv)  # 应用程序对象
    yolo_train = YOLOActivate()
    yolo_train.show()  # 展示窗口
    sys.exit(app.exec_())

