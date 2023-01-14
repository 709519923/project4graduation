# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:50:54 2021

@author: surface
"""

import sys
import signal_slot
from PyQt5.QtWidgets import QApplication,QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = signal_slot.Ui_MainWindow() #调用demo里的对象实例化
    ui.setupUi(mainWindow) #将ui实例装配到主窗口上
    mainWindow.show()
    sys.exit(app.exec_()) 
    
