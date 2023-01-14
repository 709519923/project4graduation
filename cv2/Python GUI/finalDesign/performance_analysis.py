# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:53:04 2021
model2-net1
model3-net2
model1-net
@author: surface
"""
from tensorflow.keras import models, layers
from net1 import LeNet5


def load_model(model_path = './model2/my_checkpoint'):
    model = LeNet5()
    model.build(input_shape = (None,64,64,1))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
    model.load_weights(model_path)
    model.summary()
    return model

model = load_model()

#%% 准备评测数据
import pandas as pd
df = pd.read_csv("C:/Users/surface/Desktop/学习文件/毕业设计/毕业设计/chinese_mnist/chinese_mnist.csv")
df.head()
print(df.value.values.shape)

def file_path_col(df):    
    file_path = f"input_{df[0]}_{df[1]}_{df[2]}.jpg" #input_1_1_10.jpg    
    return file_path

df["file_path"] = df.apply(file_path_col, axis = 1)

from sklearn.model_selection import train_test_split#
train_df, val_df = train_test_split(df, test_size = 0.2, random_state = 42, shuffle = True, stratify = df.code.values) 
print(train_df.shape[0])
print(val_df.shape[0])

import skimage.io
import skimage.transform
import numpy as np
file_paths = list(df.file_path)
def read_image(file_paths):
    image = skimage.io.imread("C:/Users/surface/Desktop/学习文件/毕业设计/毕业设计/chinese_mnist/data/" + file_paths)
    image = skimage.transform.resize(image, (64, 64, 1), mode="reflect") 
    # THe mode parameter determines how the array borders are handled.    
    return image[:, :, :]

# One hot encoder, but in 15 classes
def character_encoder(df, var = "character"):
    x = np.stack(df["file_path"].apply(read_image))     #增加维度[-1,64,64,3]
    y = pd.get_dummies(df[var], drop_first = False)
    return x, y

x_train, y_train = character_encoder(train_df)
x_val, y_val = character_encoder(val_df)
print(x_train.shape, ",", y_train.shape)
print(x_val.shape, ",", y_val.shape)
#%% 混淆矩阵
y_val = y_val.to_numpy()
# y_val = y_val.argmax(axis=1)
# y_pred = y_pred.argmax(axis=1)
# y_val = y_val.reshape(3000,1)
# y_pred = y_pred.reshape(3000,1)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_val)
y_val = y_val.astype(np.int)
y_pred = y_pred.astype(np.int)
con_mat = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))
# plt.imshow(con_mat, cmap=plt.cm.Blues)

con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=5)

# === plot ===
plt.figure(figsize=(15, 15))
# sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
sns.heatmap(con_mat, annot=True, cmap='Blues', fmt = '1d')

plt.ylim(0, 15)
indices = np.arange(15)+0.5
plt.xticks(indices, ['一', '七', '万', '三', '九', '二', '五', '亿','八', '六', '十', '千', '四', '百', '零'])
plt.yticks(indices, ['一', '七', '万', '三', '九', '二', '五', '亿','八', '六', '十', '千', '四', '百', '零'])
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

#%% 查准率 & 查全率 失败，这两个参数多针对二分类,可以做
from sklearn.metrics import precision_recall_curve 
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
   # highlight the threshold, add the legend, axis label and grid
   
plot_precision_recall_vs_threshold(precisions, recalls, thresholds) 
plt.show()

#%%

import tensorflow as tf

#自定义准确率函数
def accuracy(out,target,topk=(1,)):
    #按最大需要的生成预测矩阵
    maxk = max(topk)
    size = target.shape[0]
   
    # 返回前k大的值的位置索引，概率最大的索引就是预测值
    # 比如预测概率1[0.1,0.5,0.35,0.05]，则返回前3大索引[1,2,0]
    # 预测概率2[0.3, 0.6, 0.01, 0.04]，则返回前3大索引[1, 0, 3]
    # 所以得到 [
    #         [1,2,0],
    #         [1,0,3]
    #         ]

    # 按预测输入生成预测topk矩阵
    out = tf.math.top_k(out,maxk).indices
    # 当有多个维度时，进行转置，使得第一列就是对预测1的topk位置索引，第二列。。。
    # 得到新的矩阵，第一行就是对所有元素top1的预测值索引，第二行就是top2的预测值索引
    # [
    # [1,1],
    # [2,0],
    # [0,3]
    # ]
    out = tf.transpose(out,perm = [1,0])

    #比较预测矩阵与目标矩阵
    # 目标本来是一维的[2,0]，所以要扩展成与预测一样的维度方便比较
    # [
    # [2,0],
    # [2,0],
    # [2,0]
    # ]
    target = tf.broadcast_to(target,out.shape)
    # equal比较直接返回布尔矩阵
    # [
    # [false,false],
    # [true,true],
    # [false,false]
    # ]
    pre = tf.equal(out, target)


    res = []
    #循环计算topk的预测准确率
    for k in topk:
        #将前k行转换成一维的预测结果
        # correct[:k],correct的前k行预测值
        # tf.reshape(tensor, shape, name=None)，shape=-1,指转换成一维
        # tf.cast()将布尔型转换成32位浮点型的0,1
        # top1:[]
        # top2:[0.,0.,1.,1.]
        # top3:[0.,0.,1.,1.,0.,0.]
        corr = tf.cast(tf.reshape(pre[:k], [-1]),dtype=tf.float32)
        corr_total = tf.reduce_sum(corr)
        acc = float(corr_total/size)
        res.append(acc)

    return res


# =============================================================================
# 这是示例
# #生成预测值
# out = tf.random.normal([8,5])
# #对axis=1的轴做softmax时，输出结果在横轴上和为1，满足概率要求
# out = tf.math.softmax(out,axis = 1)
# #这个只是为了打印而计算
# pre = tf.math.argmax(out,axis = 1)
# 
# #生成目标值
# target = tf.random.uniform([8],maxval=5,dtype = tf.int32)
# 
# #打印初始数据
# print("out:",out.numpy())
# print("pre:",pre.numpy())
# print("target:",target.numpy())
# 
# #计算topk预测值
# acc = accuracy(out,target,topk=(1,2,3,4,5))
# 
# #打印结果
# print(acc)
# =============================================================================

y_pred = model.predict(x_val)
x_val, y_val = character_encoder(val_df)
y_val = y_val.to_numpy().argmax(axis=1)

y_pred = tf.convert_to_tensor(y_pred) # softmax
y_val = tf.convert_to_tensor(y_val) #target,非one-hot
y_val = tf.dtypes.cast(y_val, tf.int32) 
acc = accuracy(y_pred, y_val, topk=(1,2,3,4,5))
print(acc)
#[0.9973333477973938, 0.9993333220481873, 0.9993333220481873, 0.999666690826416, 0.999666690826416]
#结论：多层网络准确率太高了，需要换到正常lenet5

#%%用于top-k的索引
import numpy as np
top_k=3
arr = np.array([1, 3, 2, 4, 5])
top_k_idx=arr.argsort()[::-1][0:top_k]
print(top_k_idx)
#[4 3 1]
# =============================================================================
# topK:选择k值，返回第k个大的数的索引
# =============================================================================
def topK(array, top_k):
    top_k_idx=array.argsort()[::-1][0:top_k]
    return array[top_k_idx]

print(topK(arr,3))
