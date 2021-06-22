import cv2,os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import math
import torch
from PIL import Image
import torchvision.transforms as trans
from util import util

def get_img(input_Path):
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_paths.append(path+'/'+filename)
    return img_paths


#构建Gabor滤波器
def build_filters():
    filters = []
    # ksize = [7,9,11,13,15,17] # gabor尺度，6个
    ksize = [3,7,11] # gabor尺度，6个
    lamda = np.pi/2.0         # 波长
    for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
        for K in range(3):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    # plt.figure(1)

    #用于绘制滤波器
    # for temp in range(len(filters)):
    #     plt.subplot(4, 6, temp + 1)
    #     plt.imshow(filters[temp])
    #     plt.axis('off')
    # plt.savefig('./filters.jpg')
    return filters

#Gabor特征提取
def getGabor(img,filters):
    res = [] #滤波结果
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum))

    #用于绘制滤波效果
    # plt.figure(2)
    # for temp in range(len(res)):
    #     plt.subplot(4,6,temp+1)
    #     plt.imshow(res[temp], cmap='gray')
    #     plt.axis('off')
    # plt.savefig('./gabor.jpg')
    return res  #返回滤波结果,结果为24幅图，按照gabor角度排列

def get_gabor_feature(tensor):
    bs_imgs = tensor.cpu().numpy()
    bs_size = bs_imgs.shape[0]
    im_sz = bs_imgs.shape[-2:]

    filters = build_filters()
    res_tensor = np.zeros([bs_size,12,im_sz[0],im_sz[1]])
    for i in range(bs_size):
        image_numpy = (np.transpose(bs_imgs[i], (1, 2, 0)) + 1) / 2.0 * 255.0
        # image_numpy = util.rgb2gray(image_numpy)
        image_numpy = image_numpy[:,:,0]
        res_tensor[i] = np.array(getGabor(image_numpy.astype(np.uint8), filters))
    # transform = trans.Compose([
    #     trans.ToPILImage(),
    #     trans.ToTensor()
    # ])
    # res_tensor = transform(res_tensor).permute(0,3,1,2)
    res_tensor = torch.from_numpy(res_tensor)
    return res_tensor




def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma    # [channel_out]
    sigma_y = sigma.float() / gamma     # element-wize division, [channel_out]

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1).cuda()
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1).cuda()
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb


class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=1, requires_grad=False):
        super(GaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding

        self.Lambda = nn.Parameter(torch.rand(channel_out), requires_grad=requires_grad)
        self.theta = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=requires_grad)
        self.psi = nn.Parameter(torch.randn(channel_out) * 0.02, requires_grad=requires_grad)
        self.sigma = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=requires_grad)
        self.gamma = nn.Parameter(torch.randn(channel_out) * 0.0, requires_grad=requires_grad)

        # self.Lambda = nn.Parameter(torch.tensor(np.pi/2.0).repeat(channel_out), requires_grad=requires_grad)
        # # self.theta = nn.Parameter(torch.tensor(np.arange(start=0, stop=np.pi * (n+1) / n, step=np.pi/(n-1))), requires_grad=requires_grad)
        # self.theta = nn.Parameter(torch.tensor(np.arange(0, np.pi, np.pi / 4)).to(torch.float32), requires_grad=requires_grad)
        #
        # self.psi = nn.Parameter(torch.tensor(0).to(torch.float32).repeat(channel_out), requires_grad=requires_grad)
        # self.sigma = nn.Parameter(torch.tensor(1.0).repeat(channel_out), requires_grad=requires_grad)
        # self.gamma = nn.Parameter(torch.tensor(0.5).repeat(channel_out), requires_grad=requires_grad)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # theta = self.sigmoid(self.theta) * math.pi * 2.0
        # gamma = 1.0 + (self.gamma * 0.5)
        # sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        # Lambda = 0.001 + (self.sigmoid(self.Lambda) * 0.999)
        # psi = self.psi

        theta = self.theta
        gamma = self.gamma
        sigma = self.sigma
        Lambda = self.Lambda
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)

        kernel = kernel.float()   # [channel_out, channel_in, kernel, kernel]

        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out

if __name__ == '__main__':
    # input_Path = './'
    # filters = build_filters()
    # img_paths = get_img(input_Path)
    # for img in img_paths:
    #     img = cv2.imread(img)
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     getGabor(img, filters)

    device = torch.device('cuda:0')
    # image = torch.randn(16, 3, 128, 128).to(device)
    img = Image.open('./0016.jpg')
    transform = trans.Compose([
        trans.ToTensor()
    ])
    img = transform(img).unsqueeze(0).to(device)

    # gabor = GaborConv2d(channel_in=3,channel_out=4,kernel_size=3,padding=1).to(device)
    # res = gabor(img)

    # filters = build_filters()
    # res = getGabor(img, filters)

    res = get_gabor_feature(img)

    # for i in range(4):
    #     output = trans.ToPILImage()(res[:,i,:,:].cpu())
    #     output.save('./gaborConv2d_%d.jpg' % i)
    print('ciao!')