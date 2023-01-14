# -*- coding: utf-8 -*-
# 用于启动GUI，编写业务逻辑也在这里
"""
Created on Fri Mar 19 10:49:07 2021

@author: surface
"""


import sys,os
import test1
from test1_2 import Image_Correction,Recognition,ImgPredict
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import matplotlib.pyplot as plt
#这里就是要编写业务逻辑类
class ImageProcess(QMainWindow, test1.Ui_MainWindow):
    def __init__(self):
        super(ImageProcess, self).__init__()
        self.setupUi(self)
        self.createDataFile('./data')
        self.pushButton.clicked.connect(self.openImage)
        self.pushButton_4.clicked.connect(self.deleteImage)
        #self.pushButton.clicked.connect(self.rotateImage)
        #self.pushButton.clicked.connect(self.dilatedImage)
        self.pushButton.clicked.connect(self.display)
        #self.imgPredict = ImgPredict()
    def openImage(self): #原图片槽函数
        # try:
            print('点击pushbutton')
            self.printf('点击pushbutton')
            #方法1：完美显示图片，并自适应大小
            # pix = QtGui.QPixmap("D:/PixivWallpaper/catavento.png")
            # self.label.setPixmap(pix)
            imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
            if imgName == '':
                return -1
            #imgName返回文件绝对路径，可以根据这个路径，调用opencv来处理
            # 如果选择取消，会返回两个空的‘’，‘’
            #切忌中文路径
            jpg = QtGui.QPixmap(imgName)#.scaled(self.label.width(), self.label.height())
            # self.label.setStyleSheet("border: 2px solid blue")
           # self.label.setScaledContents(True)
            # self.label.setPixmap(jpg)
            print(imgName)
            print(jpg)
            if self.groupBox.isChecked():
                #先纠正再识别
                hough_threshold = self.spinBox.value()
                demo = Image_Correction(imgName,hough_threshold)
                if demo == -1:
                    self.printf('识别失败，请尝试对图片作出修改')
                    self.printf('无法读取图片中的文字区域，请排查是否文字背景过于复杂，已经边框有无去除')
                    return None
                self.printf('调整角度： %f' %(demo.angle))
                demo1 = Recognition('./data/rotate.jpg')
                demo2 = ImgPredict()
                # plt.imshow(demo1.cutting_image[0])
                #plt.show()
                print('Recognition list length:')
                print(len(demo1.cutting_image))
                fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
                ax = ax.flatten()
                for i in range(len(demo1.cutting_image)):
                    self.printf('第 %d 个字是： %s' %(i+1,demo2.predict(demo1.cutting_image[i])))
                    img = demo1.cutting_image[i].reshape(64,64,3)
                    img1 = demo2.imgNarrow(img)
                    #解决中文显示问题
                    plt.rcParams['font.sans-serif']=['SimHei']
                    plt.rcParams['axes.unicode_minus'] = False
                    #解决中文显示问题
                    ax[i].set_title(demo2.predict(demo1.cutting_image[i]))
                    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
                    ax[i].imshow(img1, cmap='Greys', interpolation='nearest')
                    print('已展示')
                    # 输出预测的top3概率
                    result = demo2.predictProb(demo1.cutting_image[i])
                    for i in result:
                        print(i)                    
                    
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                plt.tight_layout()
                plt.show()
                
# =============================================================================
#                 plt.figure()
#                 plt.subplot(2,2,1)
#                 plt.imshow(demo1.cutting_image[0])
#                 plt.subplot(2,2,2)
#                 plt.imshow(demo1.cutting_image[1])
#                 plt.subplot(2,2,3)
#                 plt.imshow(demo1.cutting_image[2])
# =============================================================================
                # demo1.cutting_image[0] = demo2.imgNarrow(demo1.cutting_image[0]) #(64,64,1)
                # demo1.cutting_image[1] = demo2.imgNarrow(demo1.cutting_image[1]) #(64,64,1)
                # demo1.cutting_image[0].shape
                print(demo2.predict(demo1.cutting_image[0]))
                print(demo2.predict(demo1.cutting_image[1]))
                print(demo2.predict(demo1.cutting_image[2]))
            else:
                #直接识别
                demo1 = Recognition(imgName)
                demo2 = ImgPredict()
                for i in range(len(demo1.cutting_image)):
                   self.printf('第 %d 个字是： %s' %(i+1,demo2.predict(demo1.cutting_image[i])))
# =============================================================================
#         except:
#                 print('识别退出，请重试')
#                 self.printf('识别失败，请尝试对图片作出修改\n\n')
#                 self.printf('无法读取图片中的文字区域，请排查是否文字背景过于复杂，已经边框有无去除')
# =============================================================================
    def deleteImage(self):
        path = r'./data'
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                self.deleteImage(c_path)
            else:
                os.remove(c_path)
        self.listWidget.clear()
        self.textBrowser.clear()
        print('删除data文件夹成功')
        self.printf('删除data文件夹成功') 
# =============================================================================
#     def rotateImage(self): #旋转图片槽函数
#         img_path = './data/rotate.jpg'
#         jpg = QtGui.QPixmap(img_path)#.scaled(self.label.width(), self.label.height())
#         self.label_2.setStyleSheet("border: 2px solid blue")
#        # self.label.setScaledContents(True)
#         self.label_2.setPixmap(jpg)
#     def dilatedImage(self): #框选文字前的腐蚀图像
#         img_path = './data/dilation.jpg'
#         jpg = QtGui.QPixmap(img_path)#.scaled(self.label.width(), self.label.height())
#         self.label_3.setStyleSheet("border: 2px solid blue")
#        # self.label.setScaledContents(True)
#         self.label_3.setPixmap(jpg)
#     #def finalImage(self): #识别图像
# =============================================================================
    def display(self):
        self.listWidget.setViewMode(QListView.IconMode)
        self.listWidget.setModelColumn(1)
        self.listWidget.itemSelectionChanged.connect(self.onItemSelectionChanged)

        # slider 往空间写图片
        self.verticalSlider.valueChanged.connect(self.onSliderPosChanged)
        image_dic = self.readImage('./data/')
        for i in range(len(image_dic[0])):
            image = image_dic[0][i]
            imageName = image_dic[1][i]
            self.add_image_thumbnail(image,imageName,str(i))


    def add_image_thumbnail(self, image, frameIdx, name):
        self.listWidget.itemSelectionChanged.disconnect(self.onItemSelectionChanged)

        height, width, channels = image.shape
        print(image.shape)
        bytes_per_line = width * channels
        print(bytes_per_line)
        qImage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImage)

        item = QListWidgetItem(QIcon(pixmap), str(frameIdx) + ": " + name)
        FrameIdxRole = Qt.UserRole + 1
        item.setData(FrameIdxRole, frameIdx)

        self.listWidget.addItem(item)

        # to bottom
        # self.listWidget.scrollToBottom()
        self.listWidget.setCurrentRow(self.listWidget.count() - 1)

        print('\033[32;0m  --- add image thumbnail: {}, {} -------'.format(frameIdx, name))

        self.listWidget.itemSelectionChanged.connect(self.onItemSelectionChanged)
        # self.listWidget.it

    def resizeEvent(self, event):
        width = self.listWidget.contentsRect().width()
        self.verticalSlider.setMaximum(width)
        self.verticalSlider.setValue(width - 40)

    def onItemSelectionChanged(self):
        pass

    def onSliderPosChanged(self, value):
        self.listWidget.setIconSize(QSize(value, value))

    def readImage(self, path):
        # 读取path文件夹下所有文件的名字
        imagelist = os.listdir(path)
        # 输出文件列表
        # print(imagelist)
        image = []
        for imgname,i in zip(imagelist, range(len(imagelist))):
            if(imgname.endswith(".jpg")):
                image.append(cv2.imread(path+imgname))
                #cv2.imshow(str(imgname),image[i])
                # 每张图片的停留时间
                k = cv2.waitKey(0)
                # 通过esc键终止程序
                if k == 27:
                    break
        #cv2.destroyAllWindows()
        return image,imagelist
# =============================================================================
# textBrowser输出内容用printf
# =============================================================================
    def printf(self, mes):
            self.textBrowser.append(mes)  # 在指定的区域显示提示信息
            self.cursot = self.textBrowser.textCursor()
            self.textBrowser.moveCursor(self.cursot.End)
# =============================================================================
# 创建图片写入文件夹           
# =============================================================================
    def createDataFile(self, path):
        dirs = path
        if not os.path.exists(dirs):
            os.makedirs(dirs)  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = ImageProcess() #调用demo里的对象实例化
   # ui.setupUi(mainWindow) #将ui实例装配到主窗口上
    ui.show()
    sys.exit(app.exec_()) 
    
