_# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:14:19 2021
读取文件数组(素材，已经使用过)
@author: surface
"""
"""利用opencv读取并显示一个目录下的全部图片"""
import os
import cv2
path = './data/'
# 读取path文件夹下所有文件的名字
imagelist = os.listdir(path)
# 输出文件列表
# print(imagelist)
dict = {'name','matrix'}
dict = {['Name']: 'Zara', 'Age': 7} 
image = []
for imgname,i in zip(imagelist, range(len(imagelist))):
    if(imgname.endswith(".jpg")):
        image.append(cv2.imread(path+imgname))
        cv2.imshow(str(imgname),image[i])
        # 每张图片的停留时间
        k = cv2.waitKey(0)
        # 通过esc键终止程序
        if k == 27:
            break
cv2.destroyAllWindows()


image_dic = readImage('./data/')
def readImage(path):
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
