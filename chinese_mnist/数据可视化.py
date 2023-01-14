import pandas as pd
df = pd.read_csv("chinese_mnist.csv")
df.head()
print(df.value.values.shape)

def file_path_col(df):    
    file_path = f"input_{df[0]}_{df[1]}_{df[2]}.jpg" #input_1_1_10.jpg    
    return file_path

# Create file_path column
# apply函数用于计算各列数的和，这里直接调用某列数然后以字符形式加到后面
df["file_path"] = df.apply(file_path_col, axis = 1)

from sklearn.model_selection import train_test_split#￥本0.2
train_df, test_df = train_test_split(df, test_size = 0.9, random_state = 42, shuffle = True, stratify = df.code.values) #stratify表示code这个分类在训练集和测试集的分类当中比例一致,实际上用value的值也可以
val_df, test_df   = train_test_split(df, test_size = 0.5, random_state = 42, shuffle = True, stratify = df.code.values)

print(train_df.shape[0])
print(val_df.shape[0])
print(test_df.shape[0])

import skimage.io
import skimage.transform
import numpy as np
file_paths = list(df.file_path)
def read_image(file_paths):
    image = skimage.io.imread("data/" + file_paths)
    image = skimage.transform.resize(image, (64, 64, 1), mode="reflect") 
    # THe mode parameter determines how the array borders are handled.    
    return image[:, :, :]

# One hot encoder, but in 15 classes
def character_encoder(df, var = "character"):
    x = np.stack(df["file_path"].apply(read_image))     #增加维度[-1,64,64,3]
    y = pd.get_dummies(df[var], drop_first = False) # 对character进行one-hot编码
    return x, y

x_train, y_train = character_encoder(train_df)
x_val, y_val = character_encoder(val_df)
x_test, y_test = character_encoder(test_df)

print(x_train.shape, ",", y_train.shape)
print(x_val.shape, ",", y_val.shape)
print(x_test.shape, ",", y_test.shape)

y_train = y_train.to_numpy()
y_train = np.argmax(y_train, axis=1)

#%%
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.enable_eager_execution()
print("TensorFlow Version:\t", tf.__version__)

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


#x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train / 255.0
def draw(x_character):
    x_character = x_character
    
    train_filter = np.where(y_train == x_character)
    #train_filter = np.asarray(train_filter)
    
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
    
    ax = ax.flatten()
    
    for i in range(25):
        img = x_train[train_filter[0][i]]
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    #plt.savefig('./data1/'+str(x_character)+'.jpg')
    plt.show()
    return 0

for i in range(10):
    draw(i)

