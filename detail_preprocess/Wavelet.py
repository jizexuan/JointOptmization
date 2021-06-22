from pytorch_wavelets import DWTForward, DWTInverse
import cv2,os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import os, torchvision
from PIL import Image
from torchvision import transforms as trans

def test3():
    from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
    #J为分解的层次数,wave表示使用的变换方法
    xfm = DWTForward(J=1, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='haar')

    img = Image.open('./0016.jpg')
    transform = trans.Compose([
        trans.Grayscale(),
        trans.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    Yl, Yh = xfm(img)
    print(Yl.shape)
    print(len(Yh))
    # print(Yh[0].shape)

    for i in range(len(Yh)):
        print(Yh[i].shape)
        if i == len(Yh)-1:
            h = torch.zeros([4,3,Yh[i].size(3),Yh[i].size(3)]).float()
            h[0,:,:,:] = Yl
        else:
            h = torch.zeros([3,3,Yh[i].size(3),Yh[i].size(3)]).float()
        for j in range(3):
            if i == len(Yh)-1:
                h[j+1,:,:,:] = Yh[i][:,:,j,:,:]
            else:
                h[j,:,:,:] = Yh[i][:,:,j,:,:]
        if i == len(Yh)-1:
            img_grid = torchvision.utils.make_grid(h, 2) #一行2张图片
        else:
            img_grid = torchvision.utils.make_grid(h, 3)
        torchvision.utils.save_image(img_grid, 'img_grid_{}.jpg'.format(i))

def get_img(input_Path):
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_paths.append(path+'/'+filename)
    return img_paths

def getWave(img):
    img = Image.open('./0016.jpg')
    transform = trans.Compose([
        trans.ToTensor()
    ])
    img = transform(img).unsqueeze(0)

    # img = torch.Tensor(img)
    # img = img.unsqueeze(0)

    dwt = DWTForward(J=1, wave='haar')
    if img.shape.__len__() < 4:
        img = img.unsqueeze(3)
        img = img.permute(0,3,1,2)
        Yl, Yh = dwt(img)
        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.imshow(Yl.squeeze(), cmap=plt.cm.gray)
        plt.subplot(2, 2, 2)
        plt.imshow(Yh[0].data[:, :, 0, :, :].squeeze(), cmap=plt.cm.gray)
        plt.subplot(2, 2, 3)
        plt.imshow(Yh[0].data[:, :, 1, :, :].squeeze(), cmap=plt.cm.gray)
        plt.subplot(2, 2, 4)
        plt.imshow(Yh[0].data[:, :, 2, :, :].squeeze(), cmap=plt.cm.gray)

        plt.savefig('./wavelets_gray.jpg')
    else:
        # img = img.permute(0, 3, 1, 2)
        Yl, Yh = dwt(img)
        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.imshow(Yl.squeeze().permute(1, 2, 0))
        plt.subplot(2, 2, 2)
        plt.imshow(Yh[0].data[:, :, 0, :, :].squeeze().permute(1, 2, 0))
        plt.subplot(2, 2, 3)
        plt.imshow(Yh[0].data[:, :, 1, :, :].squeeze().permute(1, 2, 0))
        plt.subplot(2, 2, 4)
        plt.imshow(Yh[0].data[:, :, 2, :, :].squeeze().permute(1, 2, 0))

        plt.savefig('./wavelets.jpg')

def get_wavelet_feature(tensor):
    img256 = F.interpolate(tensor, scale_factor=math.pow(2, 1), mode='nearest')
    dwt = DWTForward(J=1, wave='haar').cuda()
    Yl, Yh = dwt(img256)
    feature = torch.cat((Yl,Yh[0].data[:, :, 0, :, :],Yh[0].data[:, :, 1, :, :],Yh[0].data[:, :, 2, :, :]),1)
    return feature

if __name__ == '__main__':
    # input_Path = './'
    # img_paths = get_img(input_Path)
    # for img in img_paths:
    #     img = cv2.imread(img)
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     getWave(img_gray)

    test3()
    print('ciao!')