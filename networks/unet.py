# init pac unet: 1049M
# run pac unet: M
# 1bs takes 14.8125M
# total =


import torch.nn as nn
import functools
import torch
from math import sqrt
import torch.nn.functional as F
from networks.pac import PacConv2d, PacConvTranspose2d
from memory_utils.gpu_mem_track import MemTracker
import inspect
import numpy as np

class Unet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, attention=False):
        super(Unet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down
        self.downconv1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.downconv2 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downnorm2 = norm_layer(ngf * 2)

        self.downconv3 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downnorm3 = norm_layer(ngf * 4)

        self.downconv4 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downnorm4 = norm_layer(ngf * 8)

        self.downconv5 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downnorm5 = norm_layer(ngf * 8)

        self.downconv6 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downnorm6 = norm_layer(ngf * 8)

        self.downconv7 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.downrelu = nn.LeakyReLU(0.2, True)

        # innermost

        self.upconv7 = nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm7 = norm_layer(ngf * 8)

        # up
        self.upconv6 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm6 = norm_layer(ngf * 8)

        self.upconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm5 = norm_layer(ngf * 8)

        self.upconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm4 = norm_layer(ngf * 4)

        self.upconv3 = nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm3 = norm_layer(ngf * 2)

        self.upconv2 = nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm2 = norm_layer(ngf)

        self.upconv1 = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1)

        self.uprelu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            (bs, ch, bypass_m, bypass_n) = bypass.size()
            upsamp_m = upsampled.size()[2]
            upsamp_n = upsampled.size()[3]
            pad_m = (bypass.size()[2] - upsampled.size()[2]) / 2
            pad_n = (bypass.size()[3] - upsampled.size()[3]) / 2

            if pad_n != np.int(pad_n):
                a = np.int(pad_n)
                b = np.int(pad_n) + 1
            else:
                a = b = np.int(pad_n)

            if pad_m != np.int(pad_m):
                c = np.int(pad_m)
                d = np.int(pad_m) + 1
            else:
                c = d = np.int(pad_m)

            # pad = (bypass.size()[2] - upsampled.size()[2]) // 2
            # bypass = F.pad(bypass, (-pad, -pad, -pad, -pad))
            if (bypass_m < upsamp_m) | (bypass_n < upsamp_n):
                # upsampled = upsampled.resize_(bs, ch, bypass_m, bypass_n)
                upsampled = F.pad(upsampled, (a, b, c, d))

            elif (upsamp_m < bypass_m) | (upsamp_n < bypass_n):
                # bypass = bypass.resize_(bs, ch, upsamp_m, upsamp_n)
                bypass = F.pad(bypass, (-a, -b, -c, -d))

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # bs,3,128,128
        # down
        x1 = self.downconv1(x)                                   # bs,64,64,64
        x2 = self.downnorm2(self.downconv2(self.downrelu(x1)))   # bs,128,32,32
        x3 = self.downnorm3(self.downconv3(self.downrelu(x2)))   # bs,256,16,16
        x4 = self.downnorm4(self.downconv4(self.downrelu(x3)))   # bs,512,8,8
        x5 = self.downnorm5(self.downconv5(self.downrelu(x4)))   # bs,512,4,4
        x6 = self.downnorm6(self.downconv6(self.downrelu(x5)))   # bs,512,2,2
        # innermost
        x7 = self.downconv7(self.downrelu(x6))                   # bs,512,1,1
        x7 = self.upnorm7(self.upconv7(self.uprelu(x7)))         # bs,512,2,2
        x7 = torch.cat([x6, x7], 1)                              # bs,1024,2,2
        # up
        x6_ = self.upnorm6(self.upconv6(self.uprelu(x7)))        # bs,512,4,4
        x6_ = torch.cat([x5, x6_], 1)                            # bs,1024,4,4
        x5_ = self.upnorm5(self.upconv5(self.uprelu(x6_)))       # bs,512,8,8
        x5_ = torch.cat([x4, x5_], 1)                            # bs,1024,8,8
        x4_ = self.upnorm4(self.upconv4(self.uprelu(x5_)))       # bs,256,16,16
        x4_ = torch.cat([x3, x4_], 1)                            # bs,512,16,16
        x3_ = self.upnorm3(self.upconv3(self.uprelu(x4_)))       # bs,128,32,32
        x3_ = torch.cat([x2, x3_], 1)                            # bs,256,32,32
        x2_ = self.upnorm2(self.upconv2(self.uprelu(x3_)))       # bs,64,64,64
        x2_ = torch.cat([x1, x2_], 1)                            # bs,128,64,64
        x1_ = self.tanh(self.upconv1(self.uprelu(x2_)))          # bs,3,128,128


        return x1_

if __name__ == '__main__':
    device = torch.device('cuda:0')
    # frame = inspect.currentframe()  # define a frame to track
    # gpu_tracker = MemTracker(frame)  # define a GPU tracker
    # gpu_tracker.track()  # run function between th

    input_nc = output_nc = 3
    ngf = 64
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    use_dropout = True
    unet = Unet(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout).to(device)
    # gpu_tracker.track()

    image = torch.randn(1, 3, 400, 640).to(device)
    # guide = torch.randn(1,3,128,128).to(device)
    # gpu_tracker.track()

    predict = unet(image)
    # modelsize(pacunet, image)
    # gpu_tracker.track()

    # torch.cuda.empty_cache()
    # gpu_tracker.track()

    print('ciao!')