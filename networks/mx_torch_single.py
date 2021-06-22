import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        if downsample is not None:
            self.conv1_h = nn.Conv2d(inplanes, planes, kernel_size=(5, 1), stride=(2, 1),
                         padding=(2, 0), groups=1, bias=False, dilation=1)
            self.conv1_w = nn.Conv2d(inplanes, planes, kernel_size=(1, 5), stride=(1, 2),
                                     padding=(0, 2), groups=1, bias=False, dilation=1)
            self.conv2_h = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1,
                                     groups=1, bias=False, dilation=1)
            self.conv2_w = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1,
                                     groups=1, bias=False, dilation=1)
            self.conv_idt_h = nn.Conv2d(inplanes, planes, kernel_size=(5, 1), stride=(2, 1),
                         padding=(2, 0), groups=1, bias=False, dilation=1)
            self.conv_idt_w = nn.Conv2d(inplanes, planes, kernel_size=(1, 5), stride=(1, 2),
                                     padding=(0, 2), groups=1, bias=False, dilation=1)
            self.conv1_1 = nn.Conv2d(planes * 2, planes, kernel_size=1)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, list):
        out_list = []
        for i in range(len(list)):
            x = list[i]
            identity = x

            if self.downsample is not None:
                out_h = self.conv1_h(x)
                out_w = self.conv1_w(x)
                out_h = self.bn1(out_h)
                out_w = self.bn1(out_w)
                out_h = self.relu(out_h)
                out_w = self.relu(out_w)

                out_h = self.conv2_h(out_h)
                out_w = self.conv2_w(out_w)
                out_h = self.bn2(out_h)
                out_w = self.bn2(out_w)

                identity_h = self.conv_idt_h(identity)
                identity_w = self.conv_idt_w(identity)
                identity_h = self.relu(identity_h)
                identity_w = self.relu(identity_w)
                output_h = out_h + identity_h
                output_w = out_w + identity_w

                if i == 0:
                    out_list.append(output_h)
                else:
                    out_list[-1] = torch.cat((out_list[-1], output_h), 1)
                    out_list[-1] = self.conv1_1(out_list[-1])
                out_list.append(output_w)

            else:
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                out += identity
                out_list.append(self.relu(out))

        return out_list

class MX_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, channel=3):
        super(MX_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(channel, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)

        self.conv1_h = nn.Conv2d(channel, self.inplanes, kernel_size=(5, 1), stride=[2, 1], padding=[2, 0])
        self.conv1_w = nn.Conv2d(channel, self.inplanes, kernel_size=(1, 5), stride=[1, 2], padding=[0, 2])

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.maxpool_h = nn.MaxPool2d(kernel_size=[5, 1], stride=[2, 1], padding=[2, 0])
        self.maxpool_w = nn.MaxPool2d(kernel_size=[1, 5], stride=[1, 2], padding=[0, 2])
        self.conv1_1 = nn.Conv2d(self.inplanes * 2, self.inplanes, kernel_size=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_last = nn.Conv2d(512 * 6, 6, kernel_size=1, groups=6)
        self.fc = nn.Linear(6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )
            downsample = {'conv1_1' : self.conv1_1}

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x1):   # 224
        x1 = F.interpolate(x1, scale_factor=math.pow(0.5,1), mode='nearest')

        # for x1
        fmp = {'down_2x': [],
                    'down_4x': [],
                    'stack_1': [],
                    'stack_2': [],
                    'stack_3': [],
                    'stack_4': []}
        # 112
        fmp['down_2x'].append(self.conv1_h(x1))
        fmp['down_2x'].append(self.conv1_w(x1))
        fmp['down_2x'][0] = self.relu(self.bn1(fmp['down_2x'][0]))
        fmp['down_2x'][1] = self.relu(self.bn1(fmp['down_2x'][1]))
        # 56
        fmp['down_4x'].append(self.maxpool_h(fmp['down_2x'][0]))
        fmp['down_4x'].append(self.maxpool_w(fmp['down_2x'][0]))
        fmp['down_4x'][-1] = torch.cat((fmp['down_4x'][-1], self.maxpool_h(fmp['down_2x'][1])), 1)
        fmp['down_4x'][-1] = self.conv1_1(fmp['down_4x'][-1])
        fmp['down_4x'].append(self.maxpool_w(fmp['down_2x'][1]))
        # 56
        fmp['stack_1'].extend(fmp['down_4x'])
        fmp['stack_1'] = self.layer1(fmp['stack_1'])  # 56
        # 28
        fmp['stack_2'].extend(fmp['stack_1'])
        fmp['stack_2'] = self.layer2(fmp['stack_2'])  # 28
        # 14
        fmp['stack_3'].extend(fmp['stack_2'])
        fmp['stack_3'] = self.layer3(fmp['stack_3'])  # 14
        # 7
        fmp['stack_4'].extend(fmp['stack_3'])
        fmp['stack_4'] = self.layer4(fmp['stack_4'])  # 7
        # out
        out1 = []
        for i in range(len(fmp['stack_4'])):
            out1.append(torch.sum(fmp['stack_4'][i], [2, 3], keepdim=True))
        out1 = torch.cat(out1, 1)
        out1 = self.conv_last(out1)
        out1 = torch.flatten(out1, 1)
        out1 = self.fc(out1)

        output = out1
        return output

def _mx_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = MX_ResNet(block, layers, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        state_dict = torch.load('../../checkpoints/pretrained_networks/resnet18-5c106cde.pth')
        model.load_state_dict(state_dict)
    return model

def mx_resnet_single(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _mx_resnet('resnet', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                      **kwargs)

if __name__ == '__main__':
    image = torch.randn(2, 1, 256, 256)

    # model = multi_mx_resnet50(num_classes=3)
    # predict = model(image, image)

    model = mx_resnet_single(num_classes=3, channel=1)
    predict = model(image, image)
    print('ciao!')
