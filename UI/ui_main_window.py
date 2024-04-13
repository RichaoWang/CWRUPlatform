# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1250, 710)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_7 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        font = QFont()
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tw_real_time_diagnosis = QWidget()
        self.tw_real_time_diagnosis.setObjectName(u"tw_real_time_diagnosis")
        self.tw_real_time_diagnosis.setFont(font)
        self.horizontalLayout_3 = QHBoxLayout(self.tw_real_time_diagnosis)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setSpacing(10)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(10, -1, 10, -1)
        self.pb_select_model = QPushButton(self.tw_real_time_diagnosis)
        self.pb_select_model.setObjectName(u"pb_select_model")
        self.pb_select_model.setMinimumSize(QSize(0, 30))

        self.verticalLayout_3.addWidget(self.pb_select_model)


        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.gv_visual_diagnosis_data = QGraphicsView(self.tw_real_time_diagnosis)
        self.gv_visual_diagnosis_data.setObjectName(u"gv_visual_diagnosis_data")

        self.horizontalLayout.addWidget(self.gv_visual_diagnosis_data)

        self.horizontalLayout.setStretch(1, 13)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(10, -1, 10, -1)
        self.pb_local_diagnosis = QPushButton(self.tw_real_time_diagnosis)
        self.pb_local_diagnosis.setObjectName(u"pb_local_diagnosis")
        self.pb_local_diagnosis.setMinimumSize(QSize(0, 30))

        self.verticalLayout_2.addWidget(self.pb_local_diagnosis)

        self.pb_real_time_diagnosis = QPushButton(self.tw_real_time_diagnosis)
        self.pb_real_time_diagnosis.setObjectName(u"pb_real_time_diagnosis")
        self.pb_real_time_diagnosis.setMinimumSize(QSize(0, 30))

        self.verticalLayout_2.addWidget(self.pb_real_time_diagnosis)


        self.horizontalLayout_6.addLayout(self.verticalLayout_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.tb_diagnosis_result = QTextBrowser(self.tw_real_time_diagnosis)
        self.tb_diagnosis_result.setObjectName(u"tb_diagnosis_result")
        self.tb_diagnosis_result.setStyleSheet(u"border-style:solid;\n"
"border-width:1px;")

        self.horizontalLayout_2.addWidget(self.tb_diagnosis_result)

        self.horizontalLayout_2.setStretch(0, 1)

        self.horizontalLayout_6.addLayout(self.horizontalLayout_2)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.verticalLayout.setStretch(0, 3)
        self.verticalLayout.setStretch(1, 4)

        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.tabWidget.addTab(self.tw_real_time_diagnosis, "")
        self.tw_train_model = QWidget()
        self.tw_train_model.setObjectName(u"tw_train_model")
        self.horizontalLayout_8 = QHBoxLayout(self.tw_train_model)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(-1, 5, -1, 5)
        self.pb_select_file = QPushButton(self.tw_train_model)
        self.pb_select_file.setObjectName(u"pb_select_file")

        self.verticalLayout_7.addWidget(self.pb_select_file)

        self.pb_visual_data = QPushButton(self.tw_train_model)
        self.pb_visual_data.setObjectName(u"pb_visual_data")

        self.verticalLayout_7.addWidget(self.pb_visual_data)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer_2)

        self.l_select_model = QLabel(self.tw_train_model)
        self.l_select_model.setObjectName(u"l_select_model")
        font1 = QFont()
        font1.setPointSize(10)
        self.l_select_model.setFont(font1)

        self.verticalLayout_4.addWidget(self.l_select_model)

        self.comb_select_model = QComboBox(self.tw_train_model)
        self.comb_select_model.addItem("")
        self.comb_select_model.addItem("")
        self.comb_select_model.addItem("")
        self.comb_select_model.addItem("")
        self.comb_select_model.setObjectName(u"comb_select_model")

        self.verticalLayout_4.addWidget(self.comb_select_model)


        self.verticalLayout_7.addLayout(self.verticalLayout_4)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer)

        self.pb_start_training = QPushButton(self.tw_train_model)
        self.pb_start_training.setObjectName(u"pb_start_training")

        self.verticalLayout_7.addWidget(self.pb_start_training)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_5)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.l_select_show = QLabel(self.tw_train_model)
        self.l_select_show.setObjectName(u"l_select_show")
        self.l_select_show.setFont(font1)

        self.verticalLayout_5.addWidget(self.l_select_show)

        self.rb_classification_report = QRadioButton(self.tw_train_model)
        self.buttonGroup = QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName(u"buttonGroup")
        self.buttonGroup.addButton(self.rb_classification_report)
        self.rb_classification_report.setObjectName(u"rb_classification_report")
        self.rb_classification_report.setChecked(True)

        self.verticalLayout_5.addWidget(self.rb_classification_report)

        self.rb_confusion_matrix = QRadioButton(self.tw_train_model)
        self.buttonGroup.addButton(self.rb_confusion_matrix)
        self.rb_confusion_matrix.setObjectName(u"rb_confusion_matrix")

        self.verticalLayout_5.addWidget(self.rb_confusion_matrix)

        self.rb_roc = QRadioButton(self.tw_train_model)
        self.buttonGroup.addButton(self.rb_roc)
        self.rb_roc.setObjectName(u"rb_roc")

        self.verticalLayout_5.addWidget(self.rb_roc)

        self.rb_precision_recall = QRadioButton(self.tw_train_model)
        self.buttonGroup.addButton(self.rb_precision_recall)
        self.rb_precision_recall.setObjectName(u"rb_precision_recall")

        self.verticalLayout_5.addWidget(self.rb_precision_recall)

        self.rb_loss_curcv = QRadioButton(self.tw_train_model)
        self.buttonGroup.addButton(self.rb_loss_curcv)
        self.rb_loss_curcv.setObjectName(u"rb_loss_curcv")

        self.verticalLayout_5.addWidget(self.rb_loss_curcv)

        self.rb_acc_curcv = QRadioButton(self.tw_train_model)
        self.buttonGroup.addButton(self.rb_acc_curcv)
        self.rb_acc_curcv.setObjectName(u"rb_acc_curcv")

        self.verticalLayout_5.addWidget(self.rb_acc_curcv)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_3)


        self.verticalLayout_7.addLayout(self.verticalLayout_5)

        self.pb_save_model = QPushButton(self.tw_train_model)
        self.pb_save_model.setObjectName(u"pb_save_model")

        self.verticalLayout_7.addWidget(self.pb_save_model)


        self.horizontalLayout_5.addLayout(self.verticalLayout_7)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.gv_visual_data = QGraphicsView(self.tw_train_model)
        self.gv_visual_data.setObjectName(u"gv_visual_data")

        self.verticalLayout_6.addWidget(self.gv_visual_data)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.tb_train_result = QTextBrowser(self.tw_train_model)
        self.tb_train_result.setObjectName(u"tb_train_result")
        self.tb_train_result.setStyleSheet(u"border-style:solid;\n"
"border-width:1px;")

        self.horizontalLayout_4.addWidget(self.tb_train_result)

        self.l_train_result = QLabel(self.tw_train_model)
        self.l_train_result.setObjectName(u"l_train_result")
        self.l_train_result.setStyleSheet(u"background-color: rgb(255, 255, 255);\n"
"border-style:solid;\n"
"border-width:1px;")

        self.horizontalLayout_4.addWidget(self.l_train_result)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 1)

        self.verticalLayout_6.addLayout(self.horizontalLayout_4)

        self.verticalLayout_6.setStretch(0, 3)
        self.verticalLayout_6.setStretch(1, 4)

        self.horizontalLayout_5.addLayout(self.verticalLayout_6)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_5)

        self.tabWidget.addTab(self.tw_train_model, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout = QGridLayout(self.tab)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(self.tab)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.radioButton_2 = QRadioButton(self.groupBox)
        self.radioButton_2.setObjectName(u"radioButton_2")

        self.gridLayout_2.addWidget(self.radioButton_2, 4, 0, 1, 1)

        self.comboBox = QComboBox(self.groupBox)
        self.comboBox.setObjectName(u"comboBox")

        self.gridLayout_2.addWidget(self.comboBox, 2, 0, 1, 1)

        self.radioButton_3 = QRadioButton(self.groupBox)
        self.radioButton_3.setObjectName(u"radioButton_3")

        self.gridLayout_2.addWidget(self.radioButton_3, 5, 0, 1, 1)

        self.radioButton_6 = QRadioButton(self.groupBox)
        self.radioButton_6.setObjectName(u"radioButton_6")

        self.gridLayout_2.addWidget(self.radioButton_6, 8, 0, 1, 1)

        self.radioButton = QRadioButton(self.groupBox)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setChecked(True)

        self.gridLayout_2.addWidget(self.radioButton, 3, 0, 1, 1)

        self.radioButton_4 = QRadioButton(self.groupBox)
        self.radioButton_4.setObjectName(u"radioButton_4")

        self.gridLayout_2.addWidget(self.radioButton_4, 6, 0, 1, 1)

        self.radioButton_5 = QRadioButton(self.groupBox)
        self.radioButton_5.setObjectName(u"radioButton_5")

        self.gridLayout_2.addWidget(self.radioButton_5, 7, 0, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        self.label = QLabel(self.tab)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)

        self.tabWidget.addTab(self.tab, "")

        self.horizontalLayout_7.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1250, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u6eda\u52a8\u8f74\u627f\u6545\u969c\u8bca\u65ad\u5b9e\u9a8c\u53f0", None))
#if QT_CONFIG(tooltip)
        self.pb_select_model.setToolTip(QCoreApplication.translate("MainWindow", u"\u8bf7\u9009\u62e9 .m \u6216 .h5 \u683c\u5f0f\u7684\u6587\u4ef6", None))
#endif // QT_CONFIG(tooltip)
        self.pb_select_model.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u4e00\u4e2a\u6a21\u578b", None))
#if QT_CONFIG(tooltip)
        self.pb_local_diagnosis.setToolTip(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u672c\u5730\u7684\u6570\u636e\u8fdb\u884c\u8bca\u65ad", None))
#endif // QT_CONFIG(tooltip)
        self.pb_local_diagnosis.setText(QCoreApplication.translate("MainWindow", u"\u672c\u5730\u8bca\u65ad", None))
#if QT_CONFIG(tooltip)
        self.pb_real_time_diagnosis.setToolTip(QCoreApplication.translate("MainWindow", u"\u81ea\u52a8\u91c7\u96c6\u5b9e\u65f6\u4fe1\u53f7\u5e76\u8fdb\u884c\u8bca\u65ad", None))
#endif // QT_CONFIG(tooltip)
        self.pb_real_time_diagnosis.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb\u5b9e\u65f6\u8bca\u65ad", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tw_real_time_diagnosis), QCoreApplication.translate("MainWindow", u"\u5b9e\u65f6\u8bca\u65ad", None))
#if QT_CONFIG(tooltip)
        self.pb_select_file.setToolTip(QCoreApplication.translate("MainWindow", u"\u8bf7\u9009\u62e9 .mat \u683c\u5f0f\u6587\u4ef6", None))
#endif // QT_CONFIG(tooltip)
        self.pb_select_file.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u6570\u636e\u6587\u4ef6", None))
        self.pb_visual_data.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u53ef\u89c6\u5316", None))
        self.l_select_model.setText(QCoreApplication.translate("MainWindow", u"\u8bf7\u9009\u62e9\u6a21\u578b\u7c7b\u578b\uff1a", None))
        self.comb_select_model.setItemText(0, QCoreApplication.translate("MainWindow", u"1D_CNN", None))
        self.comb_select_model.setItemText(1, QCoreApplication.translate("MainWindow", u"LSTM", None))
        self.comb_select_model.setItemText(2, QCoreApplication.translate("MainWindow", u"GRU", None))
        self.comb_select_model.setItemText(3, QCoreApplication.translate("MainWindow", u"MCNN_LSTM", None))

        self.pb_start_training.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb\u8bad\u7ec3", None))
        self.l_select_show.setText(QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3\u7ed3\u679c\u5c55\u793a:", None))
        self.rb_classification_report.setText(QCoreApplication.translate("MainWindow", u"\u5206\u7c7b\u62a5\u544a", None))
        self.rb_confusion_matrix.setText(QCoreApplication.translate("MainWindow", u"\u6df7\u6dc6\u77e9\u9635", None))
        self.rb_roc.setText(QCoreApplication.translate("MainWindow", u"ROC\u66f2\u7ebf", None))
        self.rb_precision_recall.setText(QCoreApplication.translate("MainWindow", u"\u7cbe\u5ea6\u53ec\u56de\u66f2\u7ebf", None))
        self.rb_loss_curcv.setText(QCoreApplication.translate("MainWindow", u"\u635f\u5931\u66f2\u7ebf", None))
        self.rb_acc_curcv.setText(QCoreApplication.translate("MainWindow", u"\u6b63\u786e\u7387\u66f2\u7ebf", None))
        self.pb_save_model.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u6a21\u578b", None))
        self.l_train_result.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tw_train_model), QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u8bad\u7ec3", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\u5386\u53f2\u9009\u9879", None))
        self.radioButton_2.setText(QCoreApplication.translate("MainWindow", u"\u6df7\u6dc6\u77e9\u9635", None))
        self.radioButton_3.setText(QCoreApplication.translate("MainWindow", u"ROC\u66f2\u7ebf", None))
        self.radioButton_6.setText(QCoreApplication.translate("MainWindow", u"\u6b63\u786e\u7387\u66f2\u7ebf", None))
        self.radioButton.setText(QCoreApplication.translate("MainWindow", u"\u5206\u7c7b\u62a5\u544a", None))
        self.radioButton_4.setText(QCoreApplication.translate("MainWindow", u"\u7cbe\u5ea6\u53ec\u56de\u66f2\u7ebf", None))
        self.radioButton_5.setText(QCoreApplication.translate("MainWindow", u"\u635f\u5931\u66f2\u7ebf", None))
        self.label.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\u67e5\u770b\u5386\u53f2", None))
    # retranslateUi

