#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc  : 程序主入口，界面的信号槽绑定等
"""
import random

from PySide2.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QRadioButton
from PySide2.QtGui import QPixmap, QImage, QIcon
from PySide2.QtCore import Qt
from UI.ui_main_window import Ui_MainWindow

import sys
import os
from scipy.io import loadmat
from tensorflow import test
import joblib  # 这个库在安装sklearn时好像会一起安装
import threading
import numpy as np
import json

from figure_canvas import MyFigureCanvas
from data_preprocess import training_stage_prepro, diagnosis_stage_prepro
from training_model import *
from preprocess_train_result import plot_history_curcvs, plot_confusion_matrix, brief_classification_report, \
    plot_metrics
from message_signal import MyMessageSignal
from diagnosis import diagnosis, get_my_model, result_decode
from utils import generate_md5
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 给一个app图标
        self.setWindowIcon(QIcon("./UI/icon/icon.png"))
        # 定义并初始化一些成员变量
        self.data_file_path = ''  # 初始化一个 数据所在的 文件路径
        self.model_file_path = ''  # 初始化一个 模型所在的 文件路径
        self.cache_path = os.getcwd() + '/cache'  # 所有图片等的缓存路径
        self.training_flag = False  # 是否有模型在训练的标志位
        self.model_name = ''  # 初始化一个模型名字
        self.model = ''  # 训练得到的模型
        self.classification_report = ''  # 初始化一个 分类报告
        self.score = ''  # 初始化一个模型得分
        self.scaler_info = {}
        self.is_start_real_diag = False

        self.training_end_signal = MyMessageSignal()  # 训练结束信号
        self.diagnosis_end_signal = MyMessageSignal()  # # 诊断结束信号
        self.rt_training_end_signal = MyMessageSignal()  # # 诊断结束信号

        self.init_UI()

        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

    def init_UI(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.gv_visual_data_canvas = MyFigureCanvas(self.ui.gv_visual_data,
                                                    width=self.ui.gv_visual_data.width() / 10,
                                                    height=self.ui.gv_visual_data.height() / 13)
        self.gv_visual_diagnosis_data_canvas = MyFigureCanvas(self.ui.gv_visual_diagnosis_data,
                                                              width=self.ui.gv_visual_diagnosis_data.width() / 55,
                                                              height=self.ui.gv_visual_diagnosis_data.height() / 77)

        # 按钮的信号与槽连接
        self.ui.pb_select_file.clicked.connect(self.select_file)  # 模型训练页面的 选择文件 按钮
        self.ui.pb_visual_data.clicked.connect(self.visual_data)
        self.ui.pb_start_training.clicked.connect(self.start_training)  # 开始训练按钮
        # self.ui.pb_show_result.clicked.connect(self.show_result)
        self.ui.pb_save_model.clicked.connect(self.save_model)
        self.ui.pb_select_model.clicked.connect(self.select_model)
        self.ui.pb_real_time_diagnosis.clicked.connect(self.real_time_diagnosis)
        self.ui.pb_local_diagnosis.clicked.connect(self.local_diagnosis)
        # group btn 触发
        self.ui.buttonGroup.buttonClicked.connect(self.group_checked)
        self.training_end_signal.send_msg.connect(self.training_end_slot)
        self.diagnosis_end_signal.send_msg.connect(self.diagnosis_end_slot)
        self.rt_training_end_signal.send_msg.connect(self.rt_diagnosis_end_slot)

        # tab3（查看历史）的信号槽
        self.ui.tabWidget.currentChanged.connect(self.tab_change)
        self.ui.comboBox.currentIndexChanged.connect(self.cb_change)
        self.ui.radioButton.clicked.connect(self.r1checed)
        self.ui.radioButton_2.clicked.connect(self.r2checed)
        self.ui.radioButton_3.clicked.connect(self.r3checed)
        self.ui.radioButton_4.clicked.connect(self.r4checed)
        self.ui.radioButton_5.clicked.connect(self.r5checed)
        self.ui.radioButton_6.clicked.connect(self.r6checed)

    def cb_change(self):
        radioButtons = self.ui.groupBox.findChildren(QRadioButton)
        for radioButton in radioButtons:
            if radioButton.isChecked():
                radioButton.click()

    def r1checed(self):
        try:
            with open('./history/{}/classification_report.txt'.format(self.ui.comboBox.currentText()), 'r') as f:
                self.ui.label.setText(f.read())
        except Exception as e:
            pass

    def r2checed(self):
        # 读取图片文件，进行显示
        img = QImage('./history/{}/'.format(
            self.ui.comboBox.currentText()) + self.ui.comboBox.currentText() + '_confusion_matrix.png')
        img_result = img.scaled(self.ui.label.width(), self.ui.label.height(),  # 裁剪图片将图片大小
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.ui.label.setPixmap(QPixmap.fromImage(img_result))

    def r3checed(self):
        img = QImage('./history/{}/'.format(
            self.ui.comboBox.currentText()) + self.ui.comboBox.currentText() + '_ROC_Curves.png')
        img_result = img.scaled(self.ui.label.width(), self.ui.label.height(),  # 裁剪图片将图片大小
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.ui.label.setPixmap(QPixmap.fromImage(img_result))

    def r4checed(self):
        img = QImage('./history/{}/'.format(
            self.ui.comboBox.currentText()) + self.ui.comboBox.currentText() + '_Precision_Recall_Curves.png')
        img_result = img.scaled(self.ui.label.width(), self.ui.label.height(),  # 裁剪图片将图片大小
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.ui.label.setPixmap(QPixmap.fromImage(img_result))

    def r5checed(self):
        img = QImage('./history/{}/'.format(
            self.ui.comboBox.currentText()) + self.ui.comboBox.currentText() + '_train_valid_loss.png')
        img_result = img.scaled(self.ui.label.width(), self.ui.label.height(),  # 裁剪图片将图片大小
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.ui.label.setPixmap(QPixmap.fromImage(img_result))

    def r6checed(self):
        img = QImage('./history/{}/'.format(
            self.ui.comboBox.currentText()) + self.ui.comboBox.currentText() + '_train_valid_acc.png')
        img_result = img.scaled(self.ui.label.width(), self.ui.label.height(),  # 裁剪图片将图片大小
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.ui.label.setPixmap(QPixmap.fromImage(img_result))

    def tab_change(self, idx):
        self.ui.comboBox.clear()  # 清空原来的
        if idx == 2:
            # 更新combox
            his_names = os.listdir('./history')
            self.ui.comboBox.addItems(his_names)

    def group_checked(self):
        if '' == self.model_name:  # 说明还没有训练过模型
            reply = QMessageBox.information(self, '提示', '无法查看,请先训练模型！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return

        show_mode = self.ui.buttonGroup.checkedId()
        # print(show_mode)
        # !!!===== 这里的 Id 自己测出来的, 应该还有别的方法直接得到所选框的内容 ======!!!
        if -2 == show_mode:  # 展示 分类报告
            self.ui.l_train_result.setText(self.classification_report)
        elif -3 == show_mode:  # 展示 混淆矩阵
            # 读取图片文件，进行显示
            img = QImage(self.cache_path + '/' + self.model_name + '_confusion_matrix.png')
            img_result = img.scaled(self.ui.l_train_result.width(), self.ui.l_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_train_result.setPixmap(QPixmap.fromImage(img_result))
        elif -4 == show_mode:  # 展示 ROC曲线
            # 读取图片文件，进行显示
            img = QImage(self.cache_path + '/' + self.model_name + '_ROC_Curves.png')
            img_result = img.scaled(self.ui.l_train_result.width(), self.ui.l_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_train_result.setPixmap(QPixmap.fromImage(img_result))
        elif -5 == show_mode:  # 展示 精度召回曲线
            # 读取图片文件，进行显示
            img = QImage(self.cache_path + '/' + self.model_name + '_Precision_Recall_Curves.png')
            img_result = img.scaled(self.ui.l_train_result.width(), self.ui.l_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_train_result.setPixmap(QPixmap.fromImage(img_result))
        elif -6 == show_mode:  # 展示 损失曲线
            # 读取图片文件，进行显示
            img = QImage(self.cache_path + '/' + self.model_name + '_train_valid_loss.png')
            img_result = img.scaled(self.ui.l_train_result.width(), self.ui.l_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_train_result.setPixmap(QPixmap.fromImage(img_result))
        elif -7 == show_mode:  # 展示 正确率曲线
            # 读取图片文件，进行显示
            img = QImage(self.cache_path + '/' + self.model_name + '_train_valid_acc.png')
            img_result = img.scaled(self.ui.l_train_result.width(), self.ui.l_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_train_result.setPixmap(QPixmap.fromImage(img_result))

    def select_file(self):
        self.ui.pb_select_file.setEnabled(False)
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   '打开文件',  # 标题
                                                   './datas/0HP',  # 默认开始打开的文件路径， . 表示在当前文件夹下
                                                   '(*.mat)'  # 文件过滤，表示只显示后缀名为 .mat 的文件
                                                   )
        if '' != file_path:  # 选择了文件, 则将路径更新，否则，保留原路径
            self.data_file_path = file_path
            self.ui.tb_train_result.append('选择文件：' + self.data_file_path + '\n')
        self.ui.pb_select_file.setEnabled(True)

    def visual_data(self):
        self.ui.pb_visual_data.setEnabled(False)

        if '' == self.data_file_path:  # 没有选择过文件
            reply = QMessageBox.information(self, '提示', '请先选择一个数据文件！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.ui.pb_visual_data.setEnabled(True)
                return  # 直接退出

        data = read_mat(self.data_file_path)
        title = os.path.split(self.data_file_path)[-1]
        self.gv_visual_data_canvas.plot(np.arange(len(data)), data[:, 0], title=title)

        self.ui.pb_visual_data.setEnabled(True)

    def start_training(self):
        if self.training_flag:  # 有模型在训练
            reply = QMessageBox.information(self, '提示', '正在训练模型，请等待...', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return  # 退出函数

        if '' == self.data_file_path:  # 没有选择过文件
            reply = QMessageBox.information(self, '提示', '请先选择一个数据文件！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return  # 退出函数

        # 到这里，就是 没有模型在训练，且选择了文件
        # 提示用户确认
        select_model = self.ui.comb_select_model.currentText()  # 用户选择的 模型
        message = '确定使用“' + select_model + '”进行训练。\n注意：\n1.请确保所有数据在一个文件夹下！\n' \
                                               '2.请确保文件夹下有且只有用于训练的数据文件（.mat）'
        reply = QMessageBox.information(self, '提示', message, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return  # 退出函数

        # 到这里，可以开始训练了
        self.training_flag = True  # 改变标志位
        self.ui.statusbar.showMessage('正在训练模型...', 120)

        signal_length = 2048
        signal_number = 1000  # 每个类别中要抽取的样本的数量
        normal = True  # 是否标准化
        rate = [0.7, 0.2, 0.1]  # 训练集、测试集、验证集划分比例
        # 获得数据所在的文件夹  E:/fff/hhh/iii/tt.mat --> tt.mat --> E:/fff/hhh/iii/
        data_path = self.data_file_path.split('/')[-1]  # 先获得文件名
        data_path = self.data_file_path.split(data_path)[0]  # 再去掉文件路径中的文件名

        if '1D_CNN' == select_model:
            self.model_name = '1D_CNN'
            if test.is_gpu_available():
                self.ui.tb_train_result.append('\n模型选择：1D_CNN\n检测到GPU可用\n')
            else:
                self.ui.tb_train_result.append('\n模型选择：1D_CNN\n未检测到可用GPU\n')
            self.ui.tb_train_result.append('\n正在训练模型...\n')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=CNN_1D_training,
                                               args=(data_path, signal_length, signal_number, normal, rate,
                                                     self.cache_path, self.model_name, self.training_end_signal)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()

        elif 'LSTM' == select_model:
            self.model_name = 'LSTM'
            if test.is_gpu_available():
                self.ui.tb_train_result.append('\n模型选择：LSTM\n检测到GPU可用\n')
            else:
                self.ui.tb_train_result.append('\n模型选择：LSTM\n未检测到可用GPU\n')
            self.ui.tb_train_result.append('\n正在训练模型...\n')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=LSTM_training,
                                               args=(data_path, signal_length, signal_number, normal, rate,
                                                     self.cache_path, self.model_name, self.training_end_signal)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()

        elif 'GRU' == select_model:
            self.model_name = 'GRU'
            if test.is_gpu_available():
                self.ui.tb_train_result.append('\n模型选择：GRU\n检测到GPU可用\n')
            else:
                self.ui.tb_train_result.append('\n模型选择：GRU\n未检测到可用GPU\n')
            self.ui.tb_train_result.append('\n正在训练模型...\n')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=GRU_training,
                                               args=(data_path, signal_length, signal_number, normal, rate,
                                                     self.cache_path, self.model_name, self.training_end_signal)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()
        elif 'MCNN_LSTM' == select_model:
            self.model_name = 'MCNN_LSTM'
            if not test.is_gpu_available():
                reply = QMessageBox.information(self, '警告', "MCNN_LSTM仅支持GPU训练,CPU版本暂不支持.",
                                                QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    return  # 退出函数
            self.ui.tb_train_result.append('\n模型选择：MCNN_LSTM\n检测到GPU可用\n')
            self.ui.tb_train_result.append('\n正在训练模型...\n')
            # # 创建子线程，训练模型
            training_thread = threading.Thread(target=MCNN_LSTM_training,
                                               args=(data_path, 2048, 1000, normal, rate,
                                                     self.cache_path, self.model_name, self.training_end_signal)
                                               )
            # # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()

    def training_end_slot(self, msg):
        self.model = msg['model']
        self.classification_report = msg['classification_report']
        self.score = msg['score']
        self.scaler_info = msg['scaler_info']

        QMessageBox.information(self, '提示', '训练完成！', QMessageBox.Yes, QMessageBox.Yes)
        self.ui.statusbar.close()
        self.ui.tb_train_result.append('\n训练完成，模型得分：' + self.score + '\n')
        self.ui.l_train_result.setText(self.classification_report)
        self.ui.buttonGroup.button(-2).setChecked(True)
        self.training_flag = False

    def diagnosis_end_slot(self, msg):
        pred_result = msg['pred_result']
        self.ui.tb_diagnosis_result.append('\n本地诊断结果：' + pred_result + '\n')
        self.ui.pb_local_diagnosis.setEnabled(True)
        self.ui.pb_real_time_diagnosis.setEnabled(True)
        self.ui.pb_select_model.setEnabled(True)

    def rt_diagnosis_end_slot(self, msg):
        pred_result = msg['pred_result']
        self.ui.tb_diagnosis_result.append('\n实时诊断结果：' + pred_result + '\n')

    def save_model(self):
        if '' == self.model_name:  # 说明还没有训练过模型
            reply = QMessageBox.information(self, '提示', '你还没有训练模型哦！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return

        if 'random_forest' == self.model_name:
            # save_path, _ = QFileDialog.getSaveFileName(self, '保存文件', './models' + self.model_name + '.m', '(*.m)')
            # if '' == save_path:  # 没有确定保存。这里也可以通过 变量 _ 来判断
            #     return
            # print(save_path)
            save_path = "./models/{}.m".format(self.model_name)
            joblib.dump(self.model, save_path)  # 保存模型
            # 保存配置文件
            md5 = generate_md5(save_path)
            model_config = {'mean': self.scaler_info['mean'], 'std': self.scaler_info['std'], 'md5': md5}
            with open(fr'{save_path[: -2]}.json', 'w') as jf:
                json.dump(model_config, jf)
        else:
            # save_path, _ = QFileDialog.getSaveFileName(self, '保存文件', './models' + self.model_name + '.h5', '(*.h5)')
            # if '' == save_path:  # 没有确定保存。这里也可以通过 变量 _ 来判断
            #     return
            save_path = "./models/{}.h5".format(self.model_name)

            self.model.save(save_path)
            md5 = generate_md5(save_path)
            model_config = {'mean': self.scaler_info['mean'], 'std': self.scaler_info['std'], 'md5': md5}
            with open(fr'{save_path[: -3]}.json', 'w') as jf:
                json.dump(model_config, jf)

        # 放置图表到history文件夹下，未以后查看历史模型
        os.makedirs('./history/{}'.format(self.model_name), exist_ok=True)
        # self.classification_report
        with open('./history/{}/classification_report.txt'.format(self.model_name), 'w') as f:
            f.write('{}\n'.format(self.classification_report))
        for i in os.listdir('./cache'):
            if self.model_name in i:
                shutil.copy('./cache/{}'.format(i), './history/{}'.format(self.model_name))
        QMessageBox.information(self, '提示', '模型保存成功！', QMessageBox.Yes, QMessageBox.Yes)

    def select_model(self):
        self.ui.pb_select_model.setEnabled(False)
        file_path, _ = QFileDialog.getOpenFileName(self, '选择模型', './models', '(*.m *.h5)')
        if '' != file_path:  # 选择了文件, 则将路径更新，否则，保留原路径
            self.model_file_path = file_path
            # 校验模型与配置文件的一致性
            md5 = generate_md5(self.model_file_path)
            path = self.model_file_path.split('.')[0]
            try:
                with open(f'{path}.json', 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                reply = QMessageBox.information(self, '提示',
                                                '未找到配置文件！\n请确保模型和同名配置文件在同一个文件夹下',
                                                QMessageBox.Yes, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.ui.pb_select_model.setEnabled(True)
                    return
            self.model_config = json.loads(content)
            if md5 != self.model_config['md5']:
                reply = QMessageBox.information(self, '提示', '模型与配置文件不匹配，请检查！',
                                                QMessageBox.Yes, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.ui.pb_select_model.setEnabled(True)
                    return
            self.ui.tb_diagnosis_result.append('当前选择的模型为：' + self.model_file_path + '\n')
        self.ui.pb_select_model.setEnabled(True)

    def real_time_diagnosis_thread(self, model_file_path, mean, std, rt_diagnosis_end_signal):
        if not self.is_start_real_diag:
            return
        # 定位到最后一行
        self.ui.tb_diagnosis_result.verticalScrollBar().setValue(
            self.ui.tb_diagnosis_result.verticalScrollBar().maximum())
        while True:
            if not self.is_start_real_diag:
                return
            datas = os.listdir('./datas/0HP')
            real_time_data_path = os.getcwd() + '/datas/0HP/{}'.format(datas[random.randint(0, len(datas) - 1)])
            K.clear_session()
            # 绘制数据
            m_data = read_mat(real_time_data_path)
            self.gv_visual_diagnosis_data_canvas.plot(np.arange(len(m_data)), m_data[:, 0])
            diagnosis_samples = diagnosis_stage_prepro(real_time_data_path, 2048, 1000, normal=True, mean=mean,
                                                       std=std)
            pred_result = diagnosis(diagnosis_samples, model_file_path)
            msg = {'pred_result': pred_result}
            rt_diagnosis_end_signal.send_msg.emit(msg)

    def real_time_diagnosis(self):
        if self.is_start_real_diag:
            self.is_start_real_diag = False
            self.ui.pb_real_time_diagnosis.setText("开始实时诊断")
            self.ui.pb_select_model.setEnabled(True)
            self.ui.pb_local_diagnosis.setEnabled(True)
            return
        if '' == self.model_file_path:  # 没有选择过模型
            reply = QMessageBox.information(self, '提示', '你还没有选择模型哦！', QMessageBox.Yes, QMessageBox.Yes)
            if QMessageBox.Yes == reply:
                return
        self.ui.pb_select_model.setEnabled(False)
        self.ui.pb_local_diagnosis.setEnabled(False)  # 同一时间只能进行一种诊断
        self.ui.pb_real_time_diagnosis.setText("停止实时诊断")
        self.is_start_real_diag = True
        self.ui.tb_diagnosis_result.append('\n实时诊断：正在采集数据......\n')
        # TODO: 这里通过读取指定的文件夹数据来模拟实时采集数据
        # # 开个子线程进行故障诊断
        diagnosis_thread = threading.Thread(target=self.real_time_diagnosis_thread,
                                            args=(self.model_file_path,
                                                  self.model_config['mean'], self.model_config['std'],
                                                  self.rt_training_end_signal))
        diagnosis_thread.start()

    def local_diagnosis(self):
        self.ui.pb_select_model.setEnabled(False)
        self.ui.pb_local_diagnosis.setEnabled(False)
        self.ui.pb_real_time_diagnosis.setEnabled(False)  # 同一时间只能进行一种诊断

        file_path, _ = QFileDialog.getOpenFileName(self, '选择数据', './datas/0HP', '(*.mat)')
        if '' == file_path:  # 没有选择文件，也就是退出了本地诊断
            self.ui.pb_real_time_diagnosis.setEnabled(True)
            self.ui.pb_local_diagnosis.setEnabled(True)
            return

        self.ui.tb_diagnosis_result.append('\n选择文件：' + file_path + '\n')

        if '' == self.model_file_path:  # 没有选择过模型
            reply = QMessageBox.information(self, '提示', '你还没有选择模型哦！', QMessageBox.Yes, QMessageBox.Yes)
            if QMessageBox.Yes == reply:
                self.ui.pb_real_time_diagnosis.setEnabled(True)
                self.ui.pb_local_diagnosis.setEnabled(True)
                return

        self.ui.tb_diagnosis_result.verticalScrollBar().setValue(
            self.ui.tb_diagnosis_result.verticalScrollBar().maximum())

        self.ui.tb_diagnosis_result.append('\n本地诊断：正在读取数据...\n')

        data = read_mat(file_path)
        title = os.path.split(file_path)[-1]
        self.gv_visual_diagnosis_data_canvas.plot(np.arange(len(data)), data[:, 0], title=title)

        self.ui.tb_diagnosis_result.append('\n本地诊断：正在诊断..\n')

        # 开个子线程进行故障诊断
        diagnosis_thread = threading.Thread(target=fault_diagnosis,
                                            args=(self.model_file_path, file_path,
                                                  self.model_config['mean'], self.model_config['std'],
                                                  self.diagnosis_end_signal))
        diagnosis_thread.start()

    def closeEvent(self, event):
        """
        重写关闭窗口函数：在点击关闭窗口后，将缓存文件夹下的文件全部删除
        :param event:
        :return:
        """
        file_names = os.listdir(self.cache_path)
        for file_name in file_names:
            os.remove(self.cache_path + '/' + file_name)
        self.is_start_real_diag = False
        sys.exit()


def read_mat(data_path):
    """
    可视化数据
    :param data_path: 数据路径
    :return:
    """
    file = loadmat(data_path)  # 加载文件，这里得到的文件是一个 字典
    file_keys = file.keys()
    for key in file_keys:
        if 'DE' in key:  # DE: 驱动端测得的振动数据
            global data  # 定义一个全局变量
            data = file[key][:2500]  # 截取数据的前2500个数据点进行绘图
    return data


def CNN_1D_training(data_path, signal_length, signal_number, normal, rate, save_path, model_name, training_end_signal):
    """
    训练 1D_CNN 模型
    :param data_path: 数据路径
    :param signal_length: 信号长度
    :param signal_number: 型号个数
    :param normal: 是否标准化
    :param rate: 训练集，验证集，测试集 划分比例
    :param save_path: 训练完后各种图的保存路径
    :param model_name: 模型名字
    :param training_end_signal: 信号
    :return:
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_info = training_stage_prepro(data_path, signal_length,
                                                                                            signal_number, normal, rate,
                                                                                            enhance=False)
    model, history, score = training_with_1D_CNN(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128,
                                                 epochs=20, num_classes=10)
    # print(history)
    plot_history_curcvs(history, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线
    plot_confusion_matrix(model, model_name, save_path, X_test, y_test)  # 绘制混淆矩阵
    classification_report = brief_classification_report(model, model_name, X_test, y_test)  # 计算分类报告
    plot_metrics(model, model_name, save_path, X_test, y_test)  # 绘制 召回率曲线和精确度曲线
    # sleep(3)

    # 发送信号通知主线程训练完成，让主线程发个弹窗，通知用户, 同时将模型得分发送过去以便显示
    # training_end_signal.run()
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score),
           'scaler_info': scaler_info}
    training_end_signal.send_msg.emit(msg)
    # from numba import cuda
    # device = cuda.get_current_device()
    # device.reset()


def LSTM_training(data_path, signal_length, signal_number, normal, rate, save_path, model_name, training_end_signal):
    """
    训练 LSTM 模型
    :param data_path: 数据路径
    :param signal_length: 信号长度
    :param signal_number: 型号个数
    :param normal: 是否标准化
    :param rate: 训练集，验证集，测试集 划分比例
    :param save_path: 训练完后各种图的保存路径
    :param model_name: 模型名字
    :param training_end_signal: 信号
    :return:
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_info = training_stage_prepro(data_path, signal_length,
                                                                                            signal_number, normal, rate,
                                                                                            enhance=False)
    model, history, score = training_with_LSTM(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128,
                                               epochs=60, num_classes=10)
    plot_history_curcvs(history, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线
    plot_confusion_matrix(model, model_name, save_path, X_test, y_test)  # 绘制混淆矩阵
    classification_report = brief_classification_report(model, model_name, X_test, y_test)  # 计算分类报告
    plot_metrics(model, model_name, save_path, X_test, y_test)  # 绘制 召回率曲线和精确度曲线
    # sleep(3)

    # 发送信号通知主线程训练完成，让主线程发个弹窗，通知用户
    # training_end_signal.run()
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score),
           'scaler_info': scaler_info}
    training_end_signal.send_msg.emit(msg)


def MCNN_LSTM_training(data_path, signal_length, signal_number, normal, rate, save_path, model_name,
                       training_end_signal):
    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_info = training_stage_prepro(data_path, signal_length,
                                                                                            signal_number, normal, rate,
                                                                                            enhance=False)
    model, history, score = training_with_MCNN_LSTM(X_train, y_train, X_valid, y_valid, X_test, y_test)
    plot_history_curcvs(history, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线
    plot_confusion_matrix(model, model_name, save_path, X_test, y_test)  # 绘制混淆矩阵
    classification_report = brief_classification_report(model, model_name, X_test, y_test)  # 计算分类报告
    plot_metrics(model, model_name, save_path, X_test, y_test)  # 绘制 召回率曲线和精确度曲线
    # sleep(3)

    # 发送信号通知主线程训练完成，让主线程发个弹窗，通知用户
    # training_end_signal.run()
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score),
           'scaler_info': scaler_info}
    training_end_signal.send_msg.emit(msg)
    del model


def GRU_training(data_path, signal_length, signal_number, normal, rate, save_path, model_name, training_end_signal):
    """
    训练 GRU 模型
    :param data_path: 数据路径
    :param signal_length: 信号长度
    :param signal_number: 型号个数
    :param normal: 是否标准化
    :param rate: 训练集，验证集，测试集 划分比例
    :param save_path: 训练完后各种图的保存路径
    :param model_name: 模型名字
    :param training_end_signal: 信号
    :return:
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_info = training_stage_prepro(data_path, signal_length,
                                                                                            signal_number, normal, rate,
                                                                                            enhance=False)
    model, history, score = training_with_GRU(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128,
                                              epochs=60, num_classes=10)
    plot_history_curcvs(history, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线
    plot_confusion_matrix(model, model_name, save_path, X_test, y_test)  # 绘制混淆矩阵
    classification_report = brief_classification_report(model, model_name, X_test, y_test)  # 计算分类报告
    plot_metrics(model, model_name, save_path, X_test, y_test)  # 绘制 召回率曲线和精确度曲线
    # sleep(3)

    # 发送信号通知主线程训练完成，让主线程发个弹窗，通知用户
    # training_end_signal.run()
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score),
           'scaler_info': scaler_info}
    training_end_signal.send_msg.emit(msg)



def fault_diagnosis(model_file_path, real_time_data_path, mean, std, diagnosis_end_signal):
    """
    使用模型进行故障诊断
    :param model_file_path: 模型路径
    :param real_time_data_path: 数据路径
    :param mean: z-score标准化操作时的均值
    :param std: z-score标准化操作时的标准差
    :param diagnosis_end_signal: 信号
    :return:
    """

    diagnosis_samples = diagnosis_stage_prepro(real_time_data_path, 2048, 1000, normal=True, mean=mean, std=std)
    pred_result = diagnosis(diagnosis_samples, model_file_path)

    # 诊断完成，将结果发送回去
    msg = {'pred_result': pred_result}
    diagnosis_end_signal.send_msg.emit(msg)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
