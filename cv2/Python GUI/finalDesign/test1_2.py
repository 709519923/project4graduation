# -*- coding: utf-8 -*-
# 识别算法的封装，被test1_1调用，也可以用这里的main函数单独测试
"""
Created on Fri Mar 19 14:55:03 2021

@author: surface
"""
# =============================================================================
# 这个文件夹用来存放和图像处理相关的类，便于在逻辑文件夹中调用
# =============================================================================
from net1 import LeNet5
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage
from skimage import transform
# =============================================================================
# 类名：recognition
# 类内方法：
# =============================================================================
class Recognition():
    def __init__(self,image_path):
        #读入原始图像
        #image_path = './material_library/test.jpg'
        origineImage = cv2.imread(image_path)
        origineImage = self.imgNormalization(origineImage)
        # 图像灰度化    
        #image = cv2.imread('test.jpg',0)
        image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
        cv2.imshow('recognition_gray',image)
        cv2.imwrite("./data/recognition_gray.jpg", image)
        #模糊去噪，均值平滑
        image = cv2.blur(image, (2, 2))
        cv2.imshow('  blur',image)
        cv2.imwrite("./data/recognition_blur.jpg", image)
        # 将图片二值化，阈值可以自己调整,越细的字，前面的数值越要调高
        retval, img = cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)
        cv2.imshow('binary',img)    
        cv2.imwrite("./data/recognition_binary.jpg", image)
        ## b.设置卷积核5*5
        kernel = np.ones((2,2),np.uint8)
        img = cv2.dilate(img,kernel,iterations = 5) #迭代过多会使得点归入同一行，从而连着其他字一起被框
        cv2.imshow('dilation',img)
        cv2.imwrite('./data/recognition_dilation.jpg',img)
        cv2.waitKey(0)
        Position = self.pojectionMethod_get_Image_characterPosition(img)
        if Position == -1:
            return None
        cutting_Width = self.get_cutting_Width(Position)
        Position_New = self.widthMethod_getPosition(Position,cutting_Width)
        Position_New1 = self.areaMethod_getPosition(Position_New)
        self.draw_rectangle(image_path,Position_New1)
        
    '''水平投影'''
    def getHProjection(self,image):
        hProjection = np.zeros(image.shape,np.uint8)
        #图像高与宽
        (h,w)=image.shape 
        #长度与图像高度一致的数组
        h_ = [0]*h
        #循环统计每一行白色像素的个数
        for y in range(h):
            for x in range(w):
                if image[y,x] == 255:
                    h_[y]+=1
        #绘制水平投影图像
        for y in range(h):
            for x in range(h_[y]):
                hProjection[y,x] = 255
        #cv2.imshow('hProjection2',hProjection)
        cv2.imwrite('./data/recognition_hProjection2.jpg',hProjection)
        #cv2.waitKey(0)
        return h_
    
    '''垂直投影'''
    def getVProjection(self,image):
        vProjection = np.zeros(image.shape,np.uint8);
        #图像高与宽
        (h,w) = image.shape
        #长度与图像宽度一致的数组
        w_ = [0]*w
        #循环统计每一列白色像素的个数
        for x in range(w):
            for y in range(h):
                if image[y,x] == 255:
                    w_[x]+=1
        #绘制垂直平投影图像
        for x in range(w):
            for y in range(h-w_[x],h):
                vProjection[y,x] = 255
        #cv2.imshow('vProjection',vProjection)
        cv2.imwrite('./data/recognition_vProjection.jpg',vProjection)
        #cv2.waitKey(0)
        return w_
     
    #函数要求：送入图像矩阵，返回图像的水平和垂直投影之后的定位文字位置Position
    def pojectionMethod_get_Image_characterPosition(self,image):
        img = image
        #图像高与宽
        (h,w)=img.shape 
        Position = []
        #水平投影
        H = self.getHProjection(img)
        
        start = 0
        H_Start = []
        H_End = []
        #根据水平投影获取垂直分割位置
        for i in range(len(H)):
            if H[i] > 0 and start ==0:
                H_Start.append(i)
                start = 1   #投影起点标记
            if H[i] <= 0 and start == 1:
                H_End.append(i)
                start = 0   #投影终点标记
        #分割行，分割之后再进行列分割并保存分割位置
        for i in range(len(H_Start)):
            try:
                #获取行图像
                cropImg = img[H_Start[i]:H_End[i], 0:w]   # 冒号表示这一段，范围的意思
            except IndexError:
                print('无法读取图片中的文字区域，请排查是否文字背景过于复杂，已经边框有无去除')
                return -1
            #cv2.imshow('cropImg',cropImg)
            #对行图像进行垂直投影
            W = self.getVProjection(cropImg)
            Wstart = 0
            Wend = 0
            W_Start = 0
            W_End = 0
            for j in range(len(W)):
                if W[j] > 0 and Wstart ==0:
                    W_Start =j
                    Wstart = 1
                    Wend=0
                if W[j] <= 0 and Wstart == 1:
                    W_End =j
                    Wstart = 0
                    Wend=1
                if Wend == 1:
                    Position.append([W_Start,H_Start[i],W_End,H_End[i]]) #添加框框左上角、右下角坐标
                    Wend =0    
        return Position
                 
        # 1.统计每个框框的宽度
        # 2. 小于1/15的max面积的不要，减少噪声(未完成)
        # 3. 宽度大于3的，并且小于1/2max的，让其与右边的这个合并, 其他照常输出,解决 yes
        # 4. 每次的截切宽度，必须以行切割之后每行的宽度为准，而不是全部的宽度
    
    #输入位置，返回宽度数组
    def get_cutting_Width(self,Position):
        cutting_Width = [] #用于统计每一个字符识别框的宽度
        for i in range(len(Position)):
            print("width: = %d"  %(Position[i][2] - Position[i][0]))
            cutting_Width.append(Position[i][2] - Position[i][0])
        # plt.hist(cutting_Width, bins = 100)
        # plt.show()
        return cutting_Width
    #利用宽度来筛选文字框,返回新position
    def widthMethod_getPosition(self,Position,cutting_Width):
        #cutting_Width = [] #用于统计每一个字符识别框的宽度
        Position_New = [] #字符位置数组，New表示已经加入过滤算法将偏旁之类的识别框合并
        flag = -1
        for i in range(len(Position)):
            if flag == i:
                continue
            #边框合并开始
            if cutting_Width[i] > 1/10 * max(cutting_Width) and cutting_Width[i] < 1/2 * max(cutting_Width): #这个1/2可能可以改大、小一点，如3/4或
                #这个主要是要得到i，i可以给到Position  
                #Position[0][0]是左上角，第一个索引表示第几个框框
                if i+1<len(Position):
                    Position[i][2] =  Position[i+1][2]
                Position_New.append(Position[i])
                flag = i+1
               # print('i = ' , i)
            else: 
                Position_New.append(Position[i])
               # print('i = ' , i)
        return Position_New
    
    # 计算面积函数
    def area(self, width, height):
        return width * height
    
    # 利用面积的大小来筛选文字框,返回新position
    def areaMethod_getPosition(self, Position):
        area_sum = []
        Position_new = []
        for i in range(len(Position)):
            area_sum.append(self.area(Position[i][2] - Position[i][0], Position[i][3] - Position[i][1]))
        for i in range(len(Position)):
            if area_sum[i] > 1/15 * max(area_sum):
                print("%d > %d" %(area_sum[i] , 1/15 * max(area_sum)))
                Position_new.append(Position[i])
            else:
                print('find one small area')
                print("%d < %d" %(area_sum[i] , 1/15 * max(area_sum)))
                continue
        return Position_new

    # 利用画出识别的框框
    def draw_rectangle(self,image_path,Position):
        self.cutting_image = []
        image = cv2.imread(image_path)
        image = self.imgNormalization(image)
        for m in range(len(Position)):
            cv2.rectangle(image, (Position[m][0],Position[m][1]), (Position[m][2],Position[m][3]), (255 ,0 ,0), 1)
        cv2.imshow('task achieved',image)
        cv2.imwrite('./data/recognized.jpg',image)
        image = cv2.imread(image_path)
        image = self.imgNormalization(image)
        for m in range(len(Position)):
            self.cutting_image.append(cv2.resize(image[Position[m][1]:Position[m][3],Position[m][0]:Position[m][2]],(64,64)))  # 为图片重新指定尺寸
        cv2.waitKey(0)
    # =============================================================================
    # 防止图片过大，使得噪声裁切都受到影响，将其限制尺寸
    # =============================================================================
    def imgNormalization(self,image):
        height = image.shape[0]
        width = image.shape[1]
        for i in range(10):       
            if height > 1000 or width > 1000:
                height = height/2
                width = width/2
                image = cv2.resize(image, (int(width), int(height)))
            else:
                return image
        
class Image_Correction():
    def __init__(self,image_path,hough_threshold=60):
        image = cv2.imread(image_path)
        image = self.imgNormalization(image)
       # cv2.imshow("Image", image)
        # 倾斜角度矫正
        degree = self.CalcDegree(image,hough_threshold)
        print("调整角度：", degree)
        self.rotate = self.rotateImage(image, degree) #degree就是angle
        cv2.imshow("rotate", self.rotate)
        dirs = './data'
        if not os.path.exists(dirs):
                os.makedirs(dirs)  
        cv2.imwrite("./data/rotate.jpg", self.rotate)
        # cv2.imwrite("../test/recified.png", rotate, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()         
     
    # 度数转换
    def DegreeTrans(self, theta):
        res = theta / np.pi * 180
        return res
     
    # 逆时针旋转图像degree角度（原尺寸）
    def rotateImage(self, src, degree):
        # 旋转中心为图像中心
        h, w = src.shape[:2]
        # 计算二维旋转的仿射变换矩阵
        RotateMatrix = cv2.getRotationMatrix2D((w/2.0, h/2.0), degree, 1)
        print(RotateMatrix)
        # 仿射变换，背景色填充为白色
        rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
        return rotate
     
    # 通过霍夫变换计算角度
    def CalcDegree(self, srcImage,hough_threshold=600):
        midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
        dstImage = midImage
        dstImage = cv2.GaussianBlur(dstImage,(3,3),0)
        retval, dstImage = cv2.threshold(dstImage,140,255,cv2.THRESH_BINARY_INV)
        cv2.imshow('binary',dstImage)  
        kernel = np.ones((2,2),np.uint8)
        dstImage = cv2.dilate(dstImage,kernel,iterations = 10) #迭代过多会使得点归入同一行，从而连着其他字一起被框
        cv2.imshow('dilate',dstImage)
        lineimage = srcImage.copy()
        # 通过霍夫变换检测直线
        # 第4个参数就是阈值，阈值越大，检测精度越高
        lines = cv2.HoughLines(dstImage, 1, np.pi/180, hough_threshold)
        #(dilate, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
        sum = 0
        # 依次画出每条线段
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                # print("theta:", theta, " rho:", rho)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 1000 * (-b)))
                y1 = int(round(y0 + 1000 * a))
                x2 = int(round(x0 - 1000 * (-b)))
                y2 = int(round(y0 - 1000 * a))
                # 只选角度最小的作为旋转角度
                sum += theta
                # cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.imshow("Imagelines", lineimage)
                cv2.imwrite("./data/Imagelines.jpg", lineimage)
                self.lineimage = lineimage
        # 对所有角度求平均，这样做旋转效果会更好
        average = sum / len(lines)
        self.angle = self.DegreeTrans(average) - 90
        return self.angle
    # =============================================================================
    # 防止图片过大，使得噪声裁切都受到影响，将其限制尺寸
    # =============================================================================
    def imgNormalization(self,image):
        height = image.shape[0]
        width = image.shape[1]
        for i in range(10):       
            if height > 1000 or width > 1000:
                height = height/2
                width = width/2
                image = cv2.resize(image, (int(width), int(height)))
            else:
                return image

class ImgPredict():
    def __init__(self):
        self.network = LeNet5()
        self.network.build(input_shape = (None,64,64,1))
        self.network.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
        self.network.load_weights('./model2/my_checkpoint')
        #以下这个是LeNet5测试时用的模型，需要使用时注释掉前面四行
        # self.network = tf.keras.models.load_model('C:/Users/surface/Desktop/学习文件/毕业设计/毕业设计/cv2/Python GUI/finalDesign/model4/model_lenet5.h5')
        self.network.summary()
        #cutting_image = np.array(cutting_image)
        #cutting_image[1] = inverse_color(cutting_image[1])

    def predict(self,cutting_image):
        cutting_image = self.imgNarrow(cutting_image)
        cutting_image = np.reshape(cutting_image, (1,64,64,1)).astype(np.float16)
        array = self.network.predict(cutting_image)
        print('===============================================================')
        print(array)
        # print(type(array))
        array = np.argmax(array)
        code_to_character = ['一', '七', '万', '三', '九', '二', '五', '亿',
                         '八', '六', '十', '千', '四', '百', '零']
        return code_to_character[array]
    # =============================================================================
    # 用于裁剪下来的文字缩小，提高识别率
    # =============================================================================
    def imgNarrow(self,image):
        COLOR=[255,255,255]
        image = cv2.copyMakeBorder(image,20,20,50,50,cv2.BORDER_CONSTANT,value=COLOR)
        t, image = cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)
        # image = skimage.transform.resize(image, (64, 64, 1), mode="reflect")  #归一化的图片
        # image = skimage.transform.resize(image, (64, 64, 1))  #归一化的图片
        image = cv2.resize(image,(64,64))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = image.reshape(64,64,1)
        return image
    
    # =============================================================================
    # topK:选择k值，返回第k个可能的数的索引
    # =============================================================================
    def topK(self, array, top_k):
        top_k_idx=array.argsort()[::-1][0:top_k]
        print('topK index')
        print(top_k_idx.shape)
        # print(top_k_idx[0])
        return top_k_idx
    # =============================================================================
    # predictProb: 输入数组，返回top3 list：（字，概率）--以zip打包的形式
    # =============================================================================
    def predictProb(self,cutting_image):
        cutting_image = self.imgNarrow(cutting_image)
        cutting_image = np.reshape(cutting_image, (1,64,64,1)).astype(np.float16)
        array = self.network.predict(cutting_image)
        array = array[0]
        print('array.shape')
        print(array.shape)
        code_to_character = ['一', '七', '万', '三', '九', '二', '五', '亿',
                 '八', '六', '十', '千', '四', '百', '零']
        print('===============================================================')
        character = []
        prob = []            
        idx = self.topK(array, 3)
        # print(idx)
        for i in range(3):
            character.append(code_to_character[idx[i]])
            prob.append(array[idx[i]])
        a = zip(character,prob)
        print('predictProb')
        print(a)
        return a
    
    
if __name__ == '__main__':
    #demo = Recognition('C:/Users/surface/Desktop/test.jpg')
    demo = Image_Correction('C:/Users/surface/Desktop/test.jpg',hough_threshold=150)
    #demo1 = Recognition('./data/rotate.jpg')