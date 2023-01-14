# -*- coding: utf-8 -*-
# 这里可以修改网络模型的结构，具体用到的是keras的接口，可以实现简单的网络模型
# 代码有两个网络，可以选其中一个，注释另一个
"""
Created on Wed Mar  3 14:05:48 2021

@author: 70951
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model


class LeNet5(Model):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.c1=Conv2D(6,kernel_size=(5,5),input_shape = (64,64,1),padding='same',activation='relu')
        self.d1=Dropout(rate=0.2)
        self.p1=MaxPool2D(pool_size=(2,2),strides=2)
        
        self.c2=Conv2D(16,kernel_size=(5,5),padding='same',activation='relu')
        self.p2=MaxPool2D(pool_size=(2,2),strides=2)
        		
        self.flatten =Flatten()
        self.f1 =Dense(120,activation='relu')
        self.f2 =Dense(80,activation='sigmoid')
        self.f3 =Dense(15,activation='softmax')
        
    def call(self, x):
        x = self.c1(x)
        x = self.d1(x)
        x = self.p1(x)
        
        x = self.c2(x)
        x = self.p2(x)
        
        x = self.flatten(x)
        
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y

model = LeNet5()

class LeNet5(Model):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.c1=Conv2D(64,kernel_size=(5,5),input_shape = (64,64,1),padding='same',activation='relu')
        self.d1=Dropout(rate=0.2)
        self.p1=MaxPool2D(pool_size=(2,2),strides=2)
        
        self.c2=Conv2D(64,kernel_size=(5,5),padding='same',activation='relu')
        self.p2=MaxPool2D(pool_size=(2,2),strides=2)
        
        self.c3=Conv2D(128,kernel_size=(5,5),padding='same',activation='relu')
        self.p3=MaxPool2D(pool_size=(2,2),strides=2)
        
        self.c4=Conv2D(256,kernel_size=(5,5),padding='same',activation='relu')
        self.p4=MaxPool2D(pool_size=(2,2),strides=2)
        		
        self.flatten =Flatten()
        self.f1 =Dense(1024,activation='relu')
        self.f2 =Dense(512,activation='relu')
        self.f3 =Dense(128,activation='relu')
        self.f4 =Dense(15,activation='softmax')
        
    def call(self, x):
        x = self.c1(x)
        x = self.d1(x)
        x = self.p1(x)
        
        x = self.c2(x)
        x = self.p2(x)
        
        x = self.c3(x)
        x = self.p3(x)
        
        x = self.c4(x)
        x = self.p4(x)
        x = self.flatten(x)
        
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        y = self.f4(x)
        return y

# from tensorflow.keras import models, layers
# def LeNet5_with_shape():
#     model = models.Sequential()
#     # 1st Conv layer
#     model.add(Conv2D(6,kernel_size=(5,5),input_shape = (64,64,1),padding='same',activation='relu'))
#     model.add(MaxPool2D(pool_size=(2,2),strides=2))
#     # 2nd Conv layer        
#     model.add(Conv2D(16, kernel_size = (5, 5), activation = 'relu', padding = 'same'))
#     model.add(MaxPool2D(pool_size=(2,2),strides=2))
#     model.add(layers.Flatten())
#     model.add(Dense(120,activation='sigmoid'))
#     model.add(Dense(80,activation='sigmoid'))
#     model.add(Dense(15,activation='softmax'))
#     return model

# model.summary()
# #model = LeNet5()		
# model.build(input_shape = (64,64,1))
# model.summary()
