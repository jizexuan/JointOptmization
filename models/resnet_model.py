import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import cv2
from util import util
from util.saver import save_images
from torch.nn import functional as F
import gc
import random


class ResnetModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
        #     parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        #     parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        #     parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--show_cam', type=bool, default=True, help='show cam')
            parser.add_argument('--show_acc', type=bool, default=True, help='show acc for C_A and C_B')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['C', 'A', 'B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.opt = opt

        visual_names_A = []
        visual_names_B = []
        if self.opt.show_cam:
            visual_names_A.append('cam_A')
            visual_names_B.append('cam_B')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['C']
        else:  # during test time, only load Gs
            self.model_names = ['C']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netC = networks.define_C(opt.output_nc, opt.ndf, opt.netC,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # self.criterionCE = networks.CELoss().to(self.device)
            # self.criterionCE = networks.CELoss(dataset_mode=opt.dataset_mode, n_classes=opt.n_classes,
            #                                    lambda_ce_A=self.opt.lambda_ce_A, lambda_ce_B=self.opt.lambda_ce_B,
            #                                    device=self.device).to(self.device)
            self.criterionCE = nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(itertools.chain(self.netC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

        if self.opt.netC == 'inception':
            if self.isTrain:
                self.netC.train()
            else:
                self.netC.eval()

        self.features_blobs = []

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # for n classes

        AtoB = self.opt.direction == 'AtoB'
        if self.isTrain | self.opt.test_for_traindata:
            bs = len(input['train_A_paths'])

            self.train_A_label = torch.tensor(0).expand(bs).to(self.device)
            self.test_A_label = torch.tensor(0).expand(bs).to(self.device)
            self.train_B_label = torch.tensor(1).expand(bs).to(self.device)
            self.test_B_label = torch.tensor(1).expand(bs).to(self.device)

            self.real_A = input['train_A' if AtoB else 'train_B'].to(self.device)
            self.real_B = input['train_B' if AtoB else 'train_A'].to(self.device)

            idx = list(range(0, bs * 2))
            random.shuffle(idx)
            self.real = torch.cat((self.real_A, self.real_B), 0)
            self.real = self.real[idx]
            self.train_label = torch.cat((self.train_A_label, self.train_B_label), 0)
            self.train_label = self.train_label[idx]

            # self.image_paths = input['A_paths' if AtoB else 'B_paths']
            # by Gloria for test
            self.image_paths_A = input['train_A_paths' if AtoB else 'train_B_paths']
            self.image_paths_B = input['train_B_paths' if AtoB else 'train_A_paths']

            self.test_A = input['test_A' if AtoB else 'test_B'].to(self.device)
            self.test_B = input['test_B' if AtoB else 'test_A'].to(self.device)
            # self.image_paths = input['A_paths' if AtoB else 'B_paths']
            # by Gloria for test
            self.test_image_paths_A = input['test_A_paths' if AtoB else 'test_B_paths']
            self.test_image_paths_B = input['test_B_paths' if AtoB else 'test_A_paths']
            # for n classes
            if self.opt.n_classes > 2:
                if self.opt.dataset_mode == 'uwf':
                    self.train_B_label = input['train_B_label'].to(self.device)
        else:
            bs = len(input['test_A_paths'])
            self.test_A_label = torch.tensor(0).expand(bs).to(self.device)
            self.test_B_label = torch.tensor(1).expand(bs).to(self.device)

            self.real_A = input['test_A' if AtoB else 'test_B'].to(self.device)
            self.real_B = input['test_B' if AtoB else 'test_A'].to(self.device)
            # self.image_paths = input['A_paths' if AtoB else 'B_paths']
            # by Gloria for test
            self.image_paths_A = input['test_A_paths' if AtoB else 'test_B_paths']
            self.image_paths_B = input['test_B_paths' if AtoB else 'test_A_paths']
        # for n classes
        if self.opt.n_classes > 2:
            if self.opt.dataset_mode == 'uwf':
                self.test_B_label = input['test_B_label'].to(self.device)

    def cam2img(self,cam):
        output_cam = []
        # generate the class activation maps upsample to 256x256
        size_upsample = (self.opt.crop_size, self.opt.crop_size)

        # cam = np.maximum(cam, 0)  # relu
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, size_upsample)
        output = cv2.applyColorMap(cv2.resize(cam_img, size_upsample), cv2.COLORMAP_JET)
        return output

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # class_idx = [0]
        bz, nc, h, w = feature_conv.shape
        for idx in class_idx:
            feature_conv = feature_conv.cpu().numpy()
            weight_softmax = weight_softmax.cpu().numpy()
            cam = weight_softmax[idx].dot(feature_conv[0].reshape((nc, h * w)))     # this is CAM vector
            cam = cam.reshape(h, w)     # into a 2D array
        return cam

    def hook_feature(self, module, input, output):
        # self.features_blobs.append(output.data.cpu().numpy())
        self.features_blobs.append(output.data)

    def make_hook(self):
        # hook the feature extractor
        finalconv_name = 'layer4'
        self.features_blobs = []
        self.netC._modules.get('module')._modules.get(finalconv_name).register_forward_hook(self.hook_feature)

    def show_CAM(self, features_blobs, logit, isA):
        # get the softmax weight

        params = list(self.netC.parameters())
        # weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
        weight_softmax = np.squeeze(params[-2].data)
        # h_x = F.softmax(logit[0]).data.squeeze()
        # probs, idx = h_x.sort(0, True)
        # probs = probs.cpu().numpy()
        # idx = idx.cpu().numpy()
        # generate class activation mapping for the top1 prediction
        if isA:
            cam = self.returnCAM(features_blobs[0], weight_softmax, [0])
        else:
            cam = self.returnCAM(features_blobs[0], weight_softmax, [1])
        heatmap = self.cam2img(cam)
        img = torch.stack((self.real_A[0][0],self.real_A[0][0],self.real_A[0][0]),dim=2).cpu().numpy()
        # height, width, _ = img.shape
        result = heatmap * 0.1 + img * 0.5

        # del logit, params, weight_softmax, h_x, CAMs, img, height, width, _ , heatmap

        return cam,heatmap

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

    def backward(self):
        """Calculate the loss for generators G_A and G_B"""

        if self.isTrain:
            if self.opt.netC == 'inception':
                self.logit, _ = self.netC(self.real)
            else:
                self.logit = self.netC(self.real)
        else:
            if self.opt.netC == 'inception':
                self.logit_A, _ = self.netC(self.real_A)
                self.logit_B, _ = self.netC(self.real_B)
            else:
                self.logit_A = self.netC(self.real_A)
                self.logit_B = self.netC(self.real_B)

        # if self.opt.show_cam:
        #     self.make_hook()
        #     self.cam_A_ori, self.cam_A = self.show_CAM(self.features_blobs, self.logit_A, isA=True)
        #
        #     self.make_hook()
        #     self.cam_B_ori, self.cam_B = self.show_CAM(self.features_blobs, self.logit_B, isA=False)

        # self.loss_A = self.criterionCE(self.logit_A, target_label=self.train_A_label)  # target is A
        # self.loss_B = self.criterionCE(self.logit_B, target_label=self.train_B_label)  # target is B
        self.loss_C = self.criterionCE(self.logit, self.train_label)
        self.loss_A = self.loss_C
        self.loss_B = self.loss_C
        # combined loss and calculate gradients
        # self.loss_C = self.loss_A + self.loss_B
        self.loss_C.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # G_A and G_B
        self.optimizer.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward()  # calculate gradients for G_A and G_B
        self.optimizer.step()  # update G_A and G_B's weights


    def train_acc(self):

        # size_A = np.shape(self.logit_A)[0]
        # size_B = np.shape(self.logit_B)[0]
        # total = size_A + size_B
        # labels = torch.cat((torch.full([size_A], 0), torch.full([size_B], 1)), 0).long().cuda()
        total = np.shape(self.logit)[0]

        # outputs = torch.cat((self.logit_A,self.logit_B),0)
        _, predicted = torch.max(self.logit.data, 1)     # 每一行的最大值，即batch中每个样本的最大值
        correct = predicted.eq(self.train_label.data).sum()
        self.train_C_acc = 100. * correct / total

    def test_acc(self):

        if self.isTrain:
            size_A = np.shape(self.test_A)[0]
            size_B = np.shape(self.test_B)[0]
        else:
            size_A = np.shape(self.real_A)[0]
            size_B = np.shape(self.real_B)[0]
        total = size_A + size_B
        labels = torch.cat((torch.full([size_A], 0), torch.full([size_B], 1)), 0).long().cuda()

        if self.isTrain:
            if self.opt.netC == 'inception':
                logit_A, _ = self.netC(self.test_A)
                logit_B, _ = self.netC(self.test_B)
            else:
                logit_A = self.netC(self.test_A)
                logit_B = self.netC(self.test_B)
        else:
            logit_A = self.netC(self.real_A)
            logit_B = self.netC(self.real_B)
        outputs = torch.cat((logit_A,logit_B),0)
        _, predicted = torch.max(outputs.data, 1)     # 每一行的最大值，即batch中每个样本的最大值
        correct = predicted.eq(labels.data).sum()
        self.test_C_acc = 100. * correct / total
        # return predicted.data, predicted.data, predicted.data


        if self.opt.accs_or_logits == 'accs':
            return correct, correct, correct
        elif self.opt.accs_or_logits == 'labels':
            return predicted.data, predicted.data, predicted.data
        elif self.opt.accs_or_logits == 'logits':
            return outputs, outputs, outputs
