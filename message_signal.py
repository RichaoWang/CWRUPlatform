#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc  : 用于界面交互的自定义信号类
"""

from PySide2.QtCore import QObject, Signal


class MyMessageSignal(QObject):
    send_msg = Signal(dict)

    # def run(self):
    #     self.send_msg.emit('发送信号')
