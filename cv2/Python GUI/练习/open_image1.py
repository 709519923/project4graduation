# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:42:17 2021

@author: surface
"""


import sys

from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel

from PyQt5.QtGui import QPalette, QBrush, QPixmap

 

 

 

class Example(QWidget):

    

    def __init__(self):

        super().__init__()

        

        self.initUI() #界面绘制交给InitUi方法

        

    def initUI(self):

        

        pix = QPixmap('邱淑贞.jpg')

 

        lb1 = QLabel(self)

        lb1.setGeometry(0,0,500,500)  #

        lb1.setStyleSheet("border: 2px solid red")

        lb1.setPixmap(pix)

 

        

        #设置窗口的位置和大小

        self.setGeometry(300, 300, 600, 600)  

        #设置窗口的标题

        self.setWindowTitle('Example')

        

        

        #显示窗口

        self.show()

        

        

if __name__ == '__main__':

    #创建应用程序和对象

    app = QApplication(sys.argv)

    ex = Example()

    sys.exit(app.exec_()) 
