# -*- coding: utf-8 -*-
# 倾斜校正算法单独测试
"""
Created on Tue Mar  9 14:47:33 2021

@author: surface
"""

import cv2
import numpy as np
import math

def fourier_demo():
    #1、灰度化读取文件，
    img = cv2.imread('material_library/test2.jpg',0)
    #retval, img = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
    #kernel = np.ones((2,2),np.uint8)
    #img = cv2.dilate(img,kernel,iterations = 25) #迭代过多会使得点归入同一行，从而连着其他字一起被框
    #2、图像延扩
    h, w = img.shape[:2]
    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)
    cv2.imshow('new image', nimg)

    #3、执行傅里叶变换，并过得频域图像
    f = np.fft.fft2(nimg)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift))


    #二值化
    magnitude_uint = magnitude.astype(np.uint8)
    ret, thresh = cv2.threshold(magnitude_uint, 11, 255, cv2.THRESH_BINARY)
    print(ret)


    cv2.imshow('thresh', thresh)
    print(thresh.dtype)
    #霍夫直线变换
    # lines = cv2.HoughLinesP(thresh, 2, np.pi/180,threshold=30, minLineLength=40, maxLineGap=100)
    lines = cv2.HoughLinesP(thresh, 2, np.pi/180,threshold=30, minLineLength=40, maxLineGap=100)
    print(len(lines))

    #创建一个新图像，标注直线
    lineimg = np.ones(nimg.shape,dtype=np.uint8)
    lineimg = lineimg * 255

    piThresh = np.pi/180
    pi2 = np.pi/2
    print('piThresh = %f' %(piThresh))

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if x2 - x1 == 0:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue
        else:
            print('theta = %f' %(theta))

    angle = math.atan(theta)
    print('angle = %f' %(angle))
    angle = angle * (180 / np.pi)
    print('angle = %f' %(angle))
    angle = (angle - 90)/(w/h)
    #angle = angle+90
    print('angle = %f' %(angle))

    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow('line image', lineimg)
    cv2.imshow('rotated', rotated)

fourier_demo()
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# =============================================================================
# 毕设用的是这个
# =============================================================================
# coding=utf-8
import cv2
import numpy as np
input_img_file = "material_library/test2.jpg"
 
# 度数转换
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res
 
# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w/2.0, h/2.0), degree, 1)
    print(RotateMatrix)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate
 
# 通过霍夫变换计算角度
def CalcDegree(srcImage):
    midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = midImage
    #dstImage = cv2.Canny(midImage, 50, 200, 3)
    #cv2.imshow("hough",dstImage)
    dstImage = cv2.GaussianBlur(dstImage,(3,3),0)
    retval, dstImage = cv2.threshold(dstImage,120,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('binary',dstImage)  
    kernel = np.ones((2,2),np.uint8)
    dstImage = cv2.dilate(dstImage,kernel,iterations = 10) #迭代过多会使得点归入同一行，从而连着其他字一起被框
    cv2.imshow('dilate',dstImage)
    lineimage = srcImage.copy()

    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dstImage, 1, np.pi/180, 100)
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

    # 对所有角度求平均，这样做旋转效果会更好
    average = sum / len(lines)
    angle = DegreeTrans(average) - 90
    return angle
 
if __name__ == '__main__':
    image = cv2.imread(input_img_file)
    cv2.imshow("Image", image)
    # 倾斜角度矫正
    degree = CalcDegree(image)
    print("调整角度：", degree)
    rotate = rotateImage(image, degree)
    cv2.imshow("rotate", rotate)
    # cv2.imwrite("../test/recified.png", rotate, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#https://blog.csdn.net/u013063099/article/details/81937848?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
#%%

# coding=utf-8

import cv2

import numpy as np

 

img = cv2.imread('material_library/test4.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 

gaus = cv2.GaussianBlur(gray,(3,3),0)
retval, img_2 = cv2.threshold(gaus,120,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('binary',img_2)  
kernel = np.ones((2,2),np.uint8)
dilate = cv2.dilate(img_2,kernel,iterations = 45) #迭代过多会使得点归入同一行，从而连着其他字一起被框
cv2.imshow('dilation',dilate) 
 

edges = cv2.Canny(dilate, 50, 150, apertureSize=3)
cv2.imshow('edges',edges)
 

minLineLength = 100

maxLineGap = 100 #控制线段长度，越小，图像中短的线段越多

lines = cv2.HoughLinesP(dilate, 1, np.pi / 180, 100, minLineLength, maxLineGap)
#合适参数：180，100，100，90
lines = np.reshape(lines, (lines.shape[0],-1))
 

for x1, y1, x2, y2 in lines:

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

 

cv2.imshow("hough",img)

cv2.waitKey()

cv2.destroyAllWindows()

#%%

