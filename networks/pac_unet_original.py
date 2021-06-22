import torch.nn as nn
import functools
import torch
import math
import torch.nn.functional as F
from networks.pac import PacConv2d, PacConvTranspose2d
from collections import OrderedDict
from typing import Union
from memory_utils.modelsize_estimate import modelsize
from memory_utils.gpu_mem_track import MemTracker
import inspect
from detail_preprocess.Wavelet import get_wavelet_feature

class PacUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, attention=False, guide_channels=12):
        super(PacUnetGenerator, self).__init__()

        # guidance preparation
        n_g_layers = 3
        f_sz_1 = 5
        pad = int(f_sz_1 // 2)
        self.k_ch = 1  # k_ch = 16
        factor = 8
        num_ups = int(math.log2(factor))
        n_g_filters: Union[int, tuple] = 3  # n_g_filters: Union[int, tuple] = 32
        g_bn = False

        if type(n_g_filters) == int:
            n_g_filters = (n_g_filters,) * (n_g_layers - 1)
        else:
            assert len(n_g_filters) == n_g_layers - 1

        g_layers = []
        n_g_channels = (guide_channels,) + n_g_filters + (self.k_ch * num_ups,)

        for l in range(n_g_layers):
            g_layers.append(('conv{}'.format(l + 1), nn.Conv2d(n_g_channels[l], n_g_channels[l + 1],
                                                               kernel_size=f_sz_1, padding=pad)))
            if g_bn:
                g_layers.append(('bn{}'.format(l + 1), nn.BatchNorm2d(n_g_channels[l + 1])))
            if l < n_g_layers - 1:
                g_layers.append(('relu{}'.format(l + 1), nn.ReLU()))
        self.branch_g = nn.Sequential(OrderedDict(g_layers))

        # unet
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ngf = ngf
        # bs,3,128,128
        ks = 3
        op = 1
        # down
        self.downconv1 = PacConv2d(input_nc, ngf, kernel_size=ks, stride=2, padding=1, bias=use_bias)    # bs,64,64,64

        self.downconv2 = PacConv2d(ngf, ngf * 2, kernel_size=ks, stride=2, padding=1, bias=use_bias)     # bs,128,32,32
        self.downnorm2 = norm_layer(ngf * 2)

        self.downconv3 = PacConv2d(ngf * 2, ngf * 4, kernel_size=ks, stride=2, padding=1, bias=use_bias)
        self.downnorm3 = norm_layer(ngf * 4)

        self.downconv4 = PacConv2d(ngf * 4, ngf * 8, kernel_size=ks, stride=2, padding=1, bias=use_bias)
        self.downnorm4 = norm_layer(ngf * 8)

        self.downconv5 = PacConv2d(ngf * 8, ngf * 8, kernel_size=ks, stride=2, padding=1, bias=use_bias)
        self.downnorm5 = norm_layer(ngf * 8)

        self.downconv6 = PacConv2d(ngf * 8, ngf * 8, kernel_size=ks, stride=2, padding=1, bias=use_bias)
        self.downnorm6 = norm_layer(ngf * 8)

        self.downconv7 = PacConv2d(ngf * 8, ngf * 8, kernel_size=ks, stride=2, padding=1, bias=use_bias)

        self.downrelu = nn.LeakyReLU(0.2, True)

        # innermost

        self.upconv7 = PacConvTranspose2d(ngf * 8, ngf * 8, kernel_size=ks, stride=2, padding=1, output_padding=op, bias=use_bias)
        self.upnorm7 = norm_layer(ngf * 8)

        # up
        self.upconv6 = PacConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=ks, stride=2, padding=1, output_padding=op, bias=use_bias)
        self.upnorm6 = norm_layer(ngf * 8)

        self.upconv5 = PacConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=ks, stride=2, padding=1, output_padding=op, bias=use_bias)
        self.upnorm5 = norm_layer(ngf * 8)

        self.upconv4 = PacConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=ks, stride=2, padding=1, output_padding=op, bias=use_bias)
        self.upnorm4 = norm_layer(ngf * 4)

        self.upconv3 = PacConvTranspose2d(ngf * 8, ngf * 2, kernel_size=ks, stride=2, padding=1, output_padding=op, bias=use_bias)
        self.upnorm3 = norm_layer(ngf * 2)

        self.upconv2 = PacConvTranspose2d(ngf * 4, ngf, kernel_size=ks, stride=2, padding=1, output_padding=op, bias=use_bias)
        self.upnorm2 = norm_layer(ngf)

        self.upconv1 = PacConvTranspose2d(ngf * 2, output_nc, kernel_size=ks, stride=2, padding=1, output_padding=op)

        self.uprelu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def get_texture_features(self, img):
        return get_wavelet_feature(img)

    def forward(self, x):
        # prepare guide
        guide = self.branch_g(self.get_texture_features(x))

        # bs,3,128,128
        # down
        x1 = self.downconv1(x, guide)                                  # bs,64,64,64
        x2 = self.downnorm2(self.downconv2(self.downrelu(x1), F.interpolate(guide, scale_factor=math.pow(0.5,1), mode='nearest')))   # bs,128,32,32
        x3 = self.downnorm3(self.downconv3(self.downrelu(x2), F.interpolate(guide, scale_factor=math.pow(0.5,2), mode='nearest')))   # bs,256,16,16
        x4 = self.downnorm4(self.downconv4(self.downrelu(x3), F.interpolate(guide, scale_factor=math.pow(0.5,3), mode='nearest')))   # bs,512,8,8
        x5 = self.downnorm5(self.downconv5(self.downrelu(x4), F.interpolate(guide, scale_factor=math.pow(0.5,4), mode='nearest')))    # bs,512,4,4
        x6 = self.downnorm6(self.downconv6(self.downrelu(x5), F.interpolate(guide, scale_factor=math.pow(0.5,5), mode='nearest')))    # bs,512,2,2
        # innermost
        x7 = self.downconv7(self.downrelu(x6), F.interpolate(guide, scale_factor=math.pow(0.5,6), mode='nearest'))                    # bs,512,1,1
        x7 = self.upnorm7(self.upconv7(self.uprelu(x7), F.interpolate(guide, scale_factor=math.pow(0.5,6), mode='nearest')))          # bs,512,2,2
        x7 = torch.cat([x6, x7], 1)                                       # bs,1024,2,2
        # up
        x6_ = self.upnorm6(self.upconv6(self.uprelu(x7), F.interpolate(guide, scale_factor=math.pow(0.5,5), mode='nearest')))         # bs,512,4,4
        x6_ = torch.cat([x5, x6_], 1)                                     # bs,1024,4,4
        x5_ = self.upnorm5(self.upconv5(self.uprelu(x6_), F.interpolate(guide, scale_factor=math.pow(0.5,4), mode='nearest')))        # bs,512,8,8
        x5_ = torch.cat([x4, x5_], 1)                                     # bs,1024,8,8
        x4_ = self.upnorm4(self.upconv4(self.uprelu(x5_), F.interpolate(guide, scale_factor=math.pow(0.5,3), mode='nearest')))        # bs,256,16,16
        x4_ = torch.cat([x3, x4_], 1)                                     # bs,512,16,16
        x3_ = self.upnorm3(self.upconv3(self.uprelu(x4_), F.interpolate(guide, scale_factor=math.pow(0.5,2), mode='nearest')))       # bs,128,32,32
        x3_ = torch.cat([x2, x3_], 1)                                     # bs,256,32,32
        x2_ = self.upnorm2(self.upconv2(self.uprelu(x3_), F.interpolate(guide, scale_factor=math.pow(0.5,1), mode='nearest')))       # bs,64,64,64
        x2_ = torch.cat([x1, x2_], 1)                                     # bs,128,64,64
        x1_ = self.tanh(self.upconv1(self.uprelu(x2_), guide))          # bs,3,128,128

        return x1_

if __name__ == '__main__':
    device = torch.device('cuda:0')
    frame = inspect.currentframe()  # define a frame to track
    gpu_tracker = MemTracker(frame)  # define a GPU tracker
    gpu_tracker.track()  # run function between th

    input_nc = output_nc = 3
    ngf = 64
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    use_dropout = True
    pacunet = PacUnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    gpu_tracker.track()

    image = torch.randn(16,3,128,128)
    guide = torch.randn(1,3,128,128)
    predict = pacunet(image)
    # modelsize(pacunet, image)
    gpu_tracker.track()

    torch.cuda.empty_cache()
    gpu_tracker.track()

    print('ciao!')