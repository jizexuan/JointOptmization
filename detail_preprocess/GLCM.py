# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
np.set_printoptions(suppress=True)

def glcm(arr, d_x, d_y, gray_level=16):
    '''计算并返回归一化后的灰度共生矩阵'''
    max_gray = arr.max()
    height, width = arr.shape
    arr = arr.astype(np.float64)  # 将uint8类型转换为float64，以免数据失真
    arr = arr * (gray_level - 1) // max_gray  # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小。量化后灰度值范围：0 ~ gray_level - 1
    ret = np.zeros([gray_level, gray_level])
    for j in range(height -  abs(d_y)):
        for i in range(width - abs(d_x)):  # range(width - d_x)  #注释为源代码，经评论指出错误后修改
            rows = arr[j][i].astype(int)
            cols = arr[j + d_y][i + d_x].astype(int)
            ret[rows][cols] += 1
    if d_x >= d_y:
        ret = ret / float(height * (width - 1))  # 归一化, 水平方向或垂直方向
    else:
        ret = ret / float((height - 1) * (width - 1))  # 归一化, 45度或135度方向
    return ret

def feature_computer(p,gray_level = 16):
    Con=0.0
    Eng=0.0
    Asm=0.0
    Idm=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            Asm+=p[i][j]*p[i][j]
            Idm+=p[i][j]/(1+(i-j)*(i-j))
            if p[i][j]>0.0:
                Eng+=p[i][j]*math.log(p[i][j])
    return Asm,Con,-Eng,Idm

if __name__=='__main__':
    '''归一化时分母值根据角度theta变化，0度或90度时为height * (width - 1), 45度或135度时为(height - 1) * (width - 1)'''
    fp = './0016.jpg'
    img = cv2.imread(fp)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像，uint8
    glcm_0 = glcm(img_gray, 1, 0)  # 水平方向
    # glcm_1 = glcm(img_gray, 0, 1)  # 垂直方向
    # glcm_2 = glcm(img_gray, 1, 1)  # 45度方向
    # glcm_3 = glcm(img_gray, -1, 1)  # 135度方向

    asm, con, eng, idm = feature_computer(glcm_0)
    print([asm, con, eng, idm])