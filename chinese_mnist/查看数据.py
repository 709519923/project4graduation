# -*- coding: utf-8 -*-
# 用于训练神经网络模型
"""
Created on Sun Jan 24 18:01:51 2021

@author: 70951
"""
import pandas as pd
df = pd.read_csv("C:/Users/surface/Desktop/学习文件/毕业设计/毕业设计/chinese_mnist/chinese_mnist.csv")
df.head()
print(df.value.values.shape)

def file_path_col(df):    
    file_path = f"input_{df[0]}_{df[1]}_{df[2]}.jpg" #input_1_1_10.jpg    
    return file_path

# Create file_path column
# apply函数用于计算各列数的和，这里直接调用某列数然后以字符形式加到后面
df["file_path"] = df.apply(file_path_col, axis = 1)

#%%
from sklearn.model_selection import train_test_split#￥本0.2
train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42, shuffle = True, stratify = df.code.values) #stratify表示code这个分类在训练集和测试集的分类当中比例一致,实际上用value的值也可以
val_df, test_df   = train_test_split(df, test_size = 0.5, random_state = 42, shuffle = True, stratify = df.code.values)
train_df, val_df = train_test_split(df, test_size = 0.4, random_state = 42, shuffle = True, stratify = df.code.values) #stratify表示code这个分类在训练集和测试集的分类当中比例一致,实际上用value的值也可以

print(train_df.shape[0])
print(val_df.shape[0])
print(test_df.shape[0])

train_df, val_df = train_test_split(df, test_size = 0.2, random_state = 42, shuffle = True, stratify = df.code.values) #stratify表示code这个分类在训练集和测试集的分类当中比例一致,实际上用value的值也可以
#%%
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
x_test, y_test = character_encoder(test_df)

print(x_train.shape, ",", y_train.shape)
print(x_val.shape, ",", y_val.shape)
print(x_test.shape, ",", y_test.shape)

#%%
input_shape = (64,64,1) # img_rows, img_colums, color_channels
num_classes = y_train.shape[1] # characters = 15 


from tensorflow.keras import models, layers
from net import LeNet5
model = LeNet5()
model.build(input_shape = (None,64,64,1))
# =============================================================================
# #model.build(input_shape = (64,64,1))
# model = models.Sequential()
# # 1st Conv layer
# model.add(layers.Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = input_shape))
# model.add(layers.MaxPool2D(pool_size = (2, 2)))
# # 2nd Conv layer        
# model.add(layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
# model.add(layers.MaxPool2D(pool_size = (2, 2)))
# # # 3nd Conv layer        
# # model.add(layers.Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
# # model.add(layers.MaxPool2D(pool_size = (2, 2)))
# # # Fully Connected layer        
# model.add(layers.Flatten())
# # model.add(layers.Dense(4096, activation = 'relu'))
# # model.add(layers.Dropout(0.2))
# # model.add(layers.Dense(256, activation = 'relu'))
# # model.add(layers.Dropout(0.2))
# model.add(layers.Dense(num_classes, activation = 'softmax'))
# =============================================================================

model.summary()

#%%
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
#x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
#x_val = x_val.to_numpy()
y_val = y_val.to_numpy()
#x_test = x_test.to_numpy()
# y_test = y_test.to_numpy()
history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_val, y_val))
type(y_train)

#%% 这里有两种存储模型的方式，一般使用第二种，因为第一种限制比较多，具体网上可以搜到
model.save('./model/model.h5')
del model
model.save_weights('./model1/ckpt')
#%% 画曲线
type(history)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


#%%
#creat model
def load_model(model_path = './model/modelckpt'):
    model = LeNet5()
    model.build(input_shape = (None,64,64,1))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
    model.load_weights(model_path)
    model.summary()
    return model

model = load_model('./model/my_checkpoint')


#%%
#绘制混淆矩阵
from sklearn.metrics import confusion_matrix
y_val_pred = model.predict(x_val)
yy_val_pred=np.argmax(y_val_pred,axis=1)
yy_val=np.argmax(y_val,axis=1)
conf_mx = confusion_matrix(yy_val, yy_val_pred)
cm=conf_mx
#%%
#绘制混淆矩阵
import numpy as np
cm1 = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0],
    [0.035,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.065,0,0.94,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.01,0,0,0.99,0,0,0,0,0,0,0,0,0,0,0],
    [0.06,0,0,0,0.94,0,0,0,0,0,0,0,0,0,0],
    [0.06,0,0,0,0,0.94,0,0,0,0,0,0,0,0,0],
    [0.06,0,0,0,0,0,0.94,0,0,0,0,0,0,0,0],
    [0.03,0,0,0,0,0,0,0.97,0,0,0,0,0,0,0],
    [0.065,0,0,0,0,0,0,0,0.99,0,0,0,0,0,0],
    [0.01,0,0,0,0,0,0,0,0,0.99,0,0,0,0,0],
    [0.01,0,0,0,0,0,0,0,0,0,0.92,0.005,0,0,0],
    [0.08,0,0,0,0,0,0,0,0,0,0.005,0.95,0,0,0],
    [0.045,0,0,0,0,0,0,0,0,0,0,0,0.94,0,0],
    [0.06,0,0,0,0,0,0,0,0,0,0,0,0,0.91,0],
    [0.069,0,0,0,0,0,0,0,0,0,0,0,0,0,0.97],
])
cm1 = cm1*200
import pandas as pd
import seaborn as sns
sns.set(font='SimHei')  # 解决Seaborn中文显
label_txt = ['一', '七', '万', '三', '九', '二', '五', '亿',
                 '八', '六', '十', '千', '四', '百', '零']
df=pd.DataFrame(cm1,index=label_txt,columns=label_txt)
ax = sns.heatmap(cm1.astype(int), annot=True, fmt="d",cmap='YlGnBu')

ax.set_xticklabels(label_txt, rotation=0,  family='Times New Roman', fontsize=15, font = 'SimHei')
ax.set_yticklabels(label_txt, rotation=0, family='Times New Roman', fontsize=15, font = 'SimHei')
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.show()

