# 图像汉字定位算法单独测试，具体用到190行，运行时从上到下一个个模块来运行
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
file_path = 'C:\\Users\\70951\\Desktop\\python_experiment\\cv2'
#%%
'''水平投影'''
def getHProjection(image):
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
    #cv2.waitKey(0)
    return h_

'''垂直投影'''
def getVProjection(image):
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
    #cv2.waitKey(0)
    return w_
 
#函数要求：送入图像矩阵，返回图像的水平和垂直投影之后的定位文字位置Position
def pojectionMethod_get_Image_characterPosition(image):
    img = image
    #图像高与宽
    (h,w)=img.shape 
    Position = []
    #水平投影
    H = getHProjection(img)
    
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
        #获取行图像
        cropImg = img[H_Start[i]:H_End[i], 0:w]   # 冒号表示这一段，范围的意思
        #cv2.imshow('cropImg',cropImg)
        #对行图像进行垂直投影
        W = getVProjection(cropImg)
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
def get_cutting_Width(Position):
    cutting_Width = [] #用于统计每一个字符识别框的宽度
    for i in range(len(Position)):
        print("width: = %d"  %(Position[i][2] - Position[i][0]))
        cutting_Width.append(Position[i][2] - Position[i][0])
    plt.hist(cutting_Width, bins = 100)
    plt.show()
    return cutting_Width
#利用宽度来筛选文字框,返回新position
def widthMethod_getPosition(Position,cutting_Width):
    #cutting_Width = [] #用于统计每一个字符识别框的宽度
    Position_New = [] #字符位置数组，New表示已经加入过滤算法将偏旁之类的识别框合并
    flag = -1
    for i in range(len(Position)):
        if flag == i:
            continue
        #边框合并开始
        if cutting_Width[i] > 1/10 * max(cutting_Width) and cutting_Width[i] < 1/2 * max(cutting_Width): #这个1/2可能可以改大一点，如3/4
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
def area(width, height):
    return width * height

# 利用面积的大小来筛选文字框,返回新position
def areaMethod_getPosition(Position):
    area_sum = []
    Position_new = []
    for i in range(len(Position)):
        area_sum.append(area(Position[i][2] - Position[i][0], Position[i][3] - Position[i][1]))
    for i in range(len(Position)):
        if area_sum[i] > 1/5 * max(area_sum):
            print("%d > %d" %(area_sum[i] , 1/5 * max(area_sum)))
            Position_new.append(Position[i])
        else:
            print('find one')
            continue
    return Position_new
#%%
if __name__ == "__main__":
    #读入原始图像
    image_path = 'C:/Users/surface/Desktop/test/4.1/test1.jpg'
    origineImage = cv2.imread(image_path)
    # origineImage = rotate
    # 图像灰度化    
    #image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',image)
    #模糊去噪，均值平滑
    image = cv2.blur(image, (2, 2))
    cv2.imshow('  blur',image)
    # 将图片二值化，阈值可以自己调整
    retval, img = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('binary',img)    
    ## b.设置卷积核5*5
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1) #迭代过多会使得点归入同一行，从而连着其他字一起被框
    cv2.imshow('dilation',img) 
    cv2.waitKey(0)
    Position = pojectionMethod_get_Image_characterPosition(img)
    cutting_Width = get_cutting_Width(Position)
    Position_New = widthMethod_getPosition(Position,cutting_Width)
    Position_New1 = areaMethod_getPosition(Position_New)
    
    

    cutting_image = [] #裁剪下来的图片
    #根据确定的位置分割字符
    unselected_Box_Image = cv2.imread(image_path)
    width_Box_Image = cv2.imread(image_path)
    area_Box_Image = cv2.imread(image_path)
    for m in range(len(Position)):
        cv2.rectangle(unselected_Box_Image, (Position[m][0],Position[m][1]), (Position[m][2],Position[m][3]), (255 ,0 ,0), 1)
    #利用算法过滤之后分割字符
    for m in range(len(Position_New)):
         cv2.rectangle(width_Box_Image, (Position_New[m][0],Position_New[m][1]), (Position_New[m][2],Position_New[m][3]), (0 ,0 ,255), 1)  
    for m in range(len(Position_New1)):
        cv2.rectangle(area_Box_Image, (Position_New1[m][0],Position_New1[m][1]), (Position_New1[m][2],Position_New1[m][3]), (0 ,255 ,0), 1)    
    cv2.imshow('unselected_Box_Image',unselected_Box_Image)         
    cv2.imshow('width_Box_Image',width_Box_Image)
    cv2.imshow('area_Box_Image',area_Box_Image)    
    cv2.waitKey(0)
#%%
    #写图片数据
    dirs = './cutting image'
    if not os.path.exists(dirs):
        os.makedirs(dirs)   
    for m in range(len(Position_New)):
        cutting_image.append(cv2.resize(origineImage[Position_New[m][1]:Position_New[m][3],Position_New[m][0]:Position_New[m][2]],(64,64)))  # 为图片重新指定尺寸
        cv2.imwrite('./cutting image/cutting image'+str(m)+'.jpg',cv2.resize(origineImage[Position_New[m][1]:Position_New[m][3],Position_New[m][0]:Position_New[m][2]],(64,64))) #保存图片
    cv2.waitKey(0)
#%% 
#调整图片大小,要取消边框的画，上面的程序就不能话框，或者用新的图片代替
#调用模型，开始预测
import matplotlib.pyplot as plt
cutting_image[3] = cv2.cvtColor(cutting_image[3],cv2.COLOR_BGR2GRAY)
plt.imshow(cutting_image[3],cmap ='gray')
t, cutting_image[3] = cv2.threshold(cutting_image[3],120,255,cv2.THRESH_BINARY_INV)
plt.imshow(cutting_image[3],cmap ='gray')
#cutting_image[1] = np.reshape(cutting_image[1], (64,64)).astype(np.uint8)
#%%
#图片反色
# def inverse_color(image):

#     height,width = image.shape
#     img2 = image.copy()

#     for i in range(height):
#         for j in range(width):
#             img2[i,j] = (255-image[i,j]) 
#     return img2

import tensorflow as tf
network = tf.keras.models.load_model('C:/Users/surface/Desktop/学习文件/毕业设计/毕业设计/cv2/model/model.h5')
network.summary()
#cutting_image = np.array(cutting_image)
#cutting_image[1] = inverse_color(cutting_image[1])
cutting_image[3] = np.reshape(cutting_image[3], (1,64,64,1)).astype(np.float16)
a = network.predict(cutting_image[3])
a = np.argmax(a)
code_to_character = ['一', '七', '万', '三', '九', '二', '五', '亿',
                 '八', '六', '十', '千', '四', '百', '零']
print(code_to_character[a])
type(cutting_image)

