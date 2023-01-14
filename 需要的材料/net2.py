import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model


class LeNet5(Model):
    def __init__(self):
        super(LeNet5,self).__init__()
        #layer1
        self.c1=Conv2D(64,kernel_size=(5,5),input_shape = (64,64,1),padding='same',activation='relu')
        self.d1=Dropout(rate=0.2)
        self.p1=MaxPool2D(pool_size=(2,2),strides=2)
        
        #layer2
        self.c2=Conv2D(64,kernel_size=(5,5),padding='same',activation='relu')
        self.d2=Dropout(rate=0.2)
        self.p2=MaxPool2D(pool_size=(2,2),strides=2)
        
        #layer3
        self.c3=Conv2D(128,kernel_size=(5,5),padding='same',activation='relu')
        self.d3=Dropout(rate=0.2)
        self.p3=MaxPool2D(pool_size=(2,2),strides=2)
        
        #layer4
        self.c4=Conv2D(256,kernel_size=(5,5),padding='same',activation='relu')
        self.d4=Dropout(rate=0.2)        
        self.p4=MaxPool2D(pool_size=(2,2),strides=2)

        #layer5
        self.c4=Conv2D(5112,kernel_size=(5,5),padding='same',activation='relu')
        self.d4=Dropout(rate=0.2)        
        self.p4=MaxPool2D(pool_size=(2,2),strides=2)
        
        #layer_flatten & full_connection
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
        x = self.d2(x)
        x = self.p2(x)
        
        x = self.c3(x)
        x = self.d3(x)
        x = self.p3(x)
        
        x = self.c4(x)
        x = self.d4(x)
        x = self.p4(x)
        
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        y = self.f4(x)
        return y


model = LeNet5()