# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:38:27 2021

@author: 70951
"""

import pandas as pd
df = pd.read_csv("C:\\Users\\70951\\Desktop\\python_experiment\\chinese_mnist\\chinese_mnist.csv")
df.head()
print(df.value.values.shape)

def file_path_col(df):    
    file_path = f"input_{df[0]}_{df[1]}_{df[2]}.jpg" #input_1_1_10.jpg    
    return file_path

# Create file_path column
# apply函数用于计算各列数的和，这里直接调用某列数然后以字符形式加到后面
df["file_path"] = df.apply(file_path_col, axis = 1)

#%%
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42, shuffle = True, stratify = df.code.values) #stratify表示code这个分类在训练集和测试集的分类当中比例一致,实际上用value的值也可以
val_df, test_df   = train_test_split(df, test_size = 0.5, random_state = 42, shuffle = True, stratify = df.code.values)

print(train_df.shape[0])
print(val_df.shape[0])
print(test_df.shape[0])

#%%
import skimage.io
import skimage.transform
import numpy as np
file_paths = list(df.file_path)
def read_image(file_paths):
    image = skimage.io.imread("C:\\Users\\70951\\Desktop\\python_experiment\\chinese_mnist\\data\\" + file_paths)
    image = skimage.transform.resize(image, (64, 64, 1), mode="reflect") 
    # THe mode parameter determines how the array borders are handled.    
    return image[:, :, :]

# One hot encoder, but in 15 classes
def character_encoder(df, var = "code"):
    x = np.stack(df["file_path"].apply(read_image))     #增加维度[-1,64,64,3]
    y = pd.get_dummies(df[var], drop_first = False)     #one_hot编码
    return x, y

x_train, y_train = character_encoder(train_df)
x_val, y_val = character_encoder(val_df)
x_test, y_test = character_encoder(test_df)

print(x_train.shape, ",", y_train.shape)
print(x_val.shape, ",", y_val.shape)
print(x_test.shape, ",", y_test.shape)

#%% 支持向量机,差正则化，做完这个做VGG
from sklearn.svm import SVC
x_train = x_train.reshape(-1,64*64)
y_train = y_train.to_numpy()
y_train = np.argmax(y_train ,axis = 1)
x_test = x_test.reshape(-1,64*64)
y_test = y_test.to_numpy()
y_test = np.argmax(y_test ,axis = 1)
#正则化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)


svm_clf = SVC(C=100, kernel='poly', gamma='auto', max_iter=10) #罚数让划分更加慎重
svm_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = svm_clf.predict(x_test)
accuracy_score(y_test, y_pred)

