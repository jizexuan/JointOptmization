import torch.nn as nn
import functools
import torch
import math
import torch.nn.functional as F
from networks.pac import PacConv2d, PacConvTranspose2d
from networks.pac import PacConv2d, PacConvTranspose2d


class ResPacUnet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResPacUnet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down
        self.downconv1 = DownConvBlock(input_nc, ngf, use_dropout, use_bias)

        self.downconv2 = DownConvBlock(ngf, ngf * 2, use_dropout, use_bias)
        self.downnorm2 = norm_layer(ngf * 2)

        self.downconv3 = DownConvBlock(ngf * 2, ngf * 4, use_dropout, use_bias)
        self.downnorm3 = norm_layer(ngf * 4)

        self.downconv4 = DownConvBlock(ngf * 4, ngf * 8, use_dropout, use_bias)
        self.downnorm4 = norm_layer(ngf * 8)

        self.downconv5 = DownConvBlock(ngf * 8, ngf * 8, use_dropout, use_bias)
        self.downnorm5 = norm_layer(ngf * 8)

        self.downconv6 = DownConvBlock(ngf * 8, ngf * 8, use_dropout, use_bias)
        self.downnorm6 = norm_layer(ngf * 8)

        self.downconv7 = DownConvBlock(ngf * 8, ngf * 8, use_dropout, use_bias)

        self.downrelu = nn.LeakyReLU(0.2, True)

        # innermost

        self.upconv7 = UpConvBlock(ngf * 8, ngf * 8, use_dropout, use_bias)
        self.upnorm7 = norm_layer(ngf * 8)

        # up
        self.upconv6 = UpConvBlock(ngf * 8 * 2, ngf * 8, use_dropout, use_bias)
        self.upnorm6 = norm_layer(ngf * 8)

        self.upconv5 = UpConvBlock(ngf * 8 * 2, ngf * 8, use_dropout, use_bias)
        self.upnorm5 = norm_layer(ngf * 8)

        self.upconv4 = UpConvBlock(ngf * 8 * 2, ngf * 4, use_dropout, use_bias)
        self.upnorm4 = norm_layer(ngf * 4)

        self.upconv3 = UpConvBlock(ngf * 8, ngf * 2, use_dropout, use_bias)
        self.upnorm3 = norm_layer(ngf * 2)

        self.upconv2 = UpConvBlock(ngf * 4, ngf, use_dropout, use_bias)
        self.upnorm2 = norm_layer(ngf)

        self.upconv1 = UpConvBlock(ngf * 2, output_nc, use_dropout, use_bias)

        self.uprelu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    # def downconv_block(self, in_dim, out_dim, use_dropout, use_bias):
    #     conv_block = []
    #     conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)]
    #     if use_dropout:
    #         conv_block += [nn.Dropout(0.5)]
    #
    #     conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)]
    #     if use_dropout:
    #         conv_block += [nn.Dropout(0.5)]
    #
    #     conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=use_bias)]
    #
    #     return nn.Sequential(*conv_block)
    #
    # def upconv_block(self, in_dim, out_dim, use_dropout, use_bias):
    #     conv_block = []
    #     # conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)]
    #     if use_dropout:
    #         conv_block += [nn.Dropout(0.5)]
    #
    #     conv_block += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)]
    #     if use_dropout:
    #         conv_block += [nn.Dropout(0.5)]
    #
    #     # conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=use_bias)]
    #
    #     return nn.Sequential(*conv_block)

    def forward(self, x):
        # bs,3,128,128
        # down
        x1, guide1 = self.downconv1(x)                                   # bs,64,64,64
        x2, guide2 = self.downconv2(self.downrelu(x1))   # bs,128,32,32
        x2 = self.downnorm2(x2)
        x3, guide3 = self.downconv3(self.downrelu(x2))   # bs,256,16,16
        x3 = self.downnorm3(x3)
        x4, guide4 = self.downconv4(self.downrelu(x3))   # bs,512,8,8
        x4 = self.downnorm4(x4)
        x5, guide5 = self.downconv5(self.downrelu(x4))   # bs,512,4,4
        x5 = self.downnorm5(x5)
        x6, guide6 = self.downconv6(self.downrelu(x5))   # bs,512,2,2
        x6 = self.downnorm6(x6)
        # innermost
        x7, guide7 = self.downconv7(self.downrelu(x6))                   # bs,512,1,1
        x7 = self.upnorm7(self.upconv7(self.uprelu(x7), guide6))         # bs,512,2,2
        x7 = torch.cat([x6, x7], 1)                              # bs,1024,2,2
        # up
        x6_ = self.upnorm6(self.upconv6(self.uprelu(x7), guide5))        # bs,512,4,4
        x6_ = torch.cat([x5, x6_], 1)                            # bs,1024,4,4
        x5_ = self.upnorm5(self.upconv5(self.uprelu(x6_), guide4))       # bs,512,8,8
        x5_ = torch.cat([x4, x5_], 1)                            # bs,1024,8,8
        x4_ = self.upnorm4(self.upconv4(self.uprelu(x5_), guide3))       # bs,256,16,16
        x4_ = torch.cat([x3, x4_], 1)                            # bs,512,16,16
        x3_ = self.upnorm3(self.upconv3(self.uprelu(x4_), guide2))       # bs,128,32,32
        x3_ = torch.cat([x2, x3_], 1)                            # bs,256,32,32
        x2_ = self.upnorm2(self.upconv2(self.uprelu(x3_), guide1))       # bs,64,64,64
        x2_ = torch.cat([x1, x2_], 1)                            # bs,128,64,64
        x1_ = self.tanh(self.upconv1(self.uprelu(x2_), x))          # bs,3,128,128

        return x1_

class DownConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_dropout, use_bias):
        super(DownConvBlock, self).__init__()
        conv_block = []
        conv_block += [nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        # conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.conv =  nn.Sequential(*conv_block)

        conv_block = []
        # conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        # conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.res =  nn.Sequential(*conv_block)

        self.pac = PacConv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)    # bs,64,64,64

    def forward(self, x):
        guidance = self.res(x)
        return self.pac(self.conv(x), guidance), guidance

class UpConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_dropout, use_bias):
        super(UpConvBlock, self).__init__()
        conv_block = []
        # conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        # conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.conv =  nn.Sequential(*conv_block)

        conv_block = []
        # conv_block += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        # conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.res =  nn.Sequential(*conv_block)

        self.pac = PacConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)    # bs,64,64,64

    def forward(self, x, guide):
        # return self.pac(self.conv(x), self.res(x))
        return self.pac(self.conv(x), guide)

if __name__ == '__main__':
    device = torch.device('cuda:0')

    input_nc = output_nc = 3
    ngf = 64
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    use_dropout = True
    net = ResPacUnet(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout).to(device)

    image = torch.randn(1,3,128,128).to(device)

    predict = net(image).to(device)

    print('ciao!')
