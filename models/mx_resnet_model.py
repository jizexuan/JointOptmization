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


class MXResnetModel(BaseModel):
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
        parser.add_argument('--n_classes', type=int, default=2, help='how many class')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['C', 'C_A', 'C_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.opt = opt
        self.only_early_late = 'onlyEarlyLate' in self.opt.name

        visual_names_A = []
        visual_names_B = []
        if self.opt.show_cam:
            visual_names_A.append('cam_A')
            visual_names_B.append('cam_B')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = []
        self.model_names.append('C')

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netC = networks.define_C(1, opt.ndf, opt.netC,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                        opt.n_classes)

        if self.isTrain:
            # self.criterionCE = networks.CELoss().to(self.device)
            # self.criterionCE = networks.CELoss(dataset_mode=opt.dataset_mode, n_classes=opt.n_classes,
            #                                    lambda_ce_A=self.opt.lambda_ce_A, lambda_ce_B=self.opt.lambda_ce_B,
            #                                    device=self.device).to(self.device)
            self.criterionCE = networks.CELoss(dataset_mode=opt.dataset_mode, n_classes=opt.n_classes,
                                               lambda_ce_A=self.opt.lambda_ce_A, lambda_ce_B=self.opt.lambda_ce_B,
                                               device=self.device).to(self.device)
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
            if not self.only_early_late:
                self.real1_A = self.real_A[:,0:3,:,:]
                self.real1_B = self.real_B[:,0:3,:,:]
            else:
                self.real2_A = self.real_A[:,0,:,:].unsqueeze(1)
                self.real2_B = self.real_B[:,0,:,:].unsqueeze(1)
                self.real3_A = self.real_A[:,1,:,:].unsqueeze(1)
                self.real3_B = self.real_B[:,1,:,:].unsqueeze(1)

            # self.image_paths = input['A_paths' if AtoB else 'B_paths']
            # by Gloria for test
            self.image_paths_A = input['train_A_paths' if AtoB else 'train_B_paths']
            self.image_paths_B = input['train_B_paths' if AtoB else 'train_A_paths']

            self.test_A = input['test_A' if AtoB else 'test_B'].to(self.device)
            self.test_B = input['test_B' if AtoB else 'test_A'].to(self.device)
            if not self.only_early_late:
                self.test1_A = self.test_A[:,0:3,:,:]
                self.test1_B = self.test_B[:,0:3,:,:]
            else:
                self.test2_A = self.test_A[:,0,:,:].unsqueeze(1)
                self.test2_B = self.test_B[:,0,:,:].unsqueeze(1)
                self.test3_A = self.test_A[:,0,:,:].unsqueeze(1)
                self.test3_B = self.test_B[:,0,:,:].unsqueeze(1)
            # self.image_paths = input['A_paths' if AtoB else 'B_paths']
            # by Gloria for test
            self.test_image_paths_A = input['test_A_paths' if AtoB else 'test_B_paths']
            self.test_image_paths_B = input['test_B_paths' if AtoB else 'test_A_paths']
            # for n classes
            if self.opt.n_classes > 2:
                self.train_B_label = input['train_B_label'].to(self.device)
        else:
            bs = len(input['test_A_paths'])
            self.test_A_label = torch.tensor(0).expand(bs).to(self.device)
            self.test_B_label = torch.tensor(1).expand(bs).to(self.device)

            self.real_A = input['test_A' if AtoB else 'test_B'].to(self.device)
            self.real_B = input['test_B' if AtoB else 'test_A'].to(self.device)
            if not self.only_early_late:
                self.real1_A = self.real_A[:, 0:3, :, :]
                self.real1_B = self.real_B[:, 0:3, :, :]
            else:
                self.real2_A = self.real_A[:,0,:,:].unsqueeze(1)
                self.real2_B = self.real_B[:,0,:,:].unsqueeze(1)
                self.real3_A = self.real_A[:,1,:,:].unsqueeze(1)
                self.real3_B = self.real_B[:,1,:,:].unsqueeze(1)
            # self.image_paths = input['A_paths' if AtoB else 'B_paths']
            # by Gloria for test
            self.image_paths_A = input['test_A_paths' if AtoB else 'test_B_paths']
            self.image_paths_B = input['test_B_paths' if AtoB else 'test_A_paths']
            # for n classes
            if self.opt.n_classes > 2:
                self.test_B_label = input['test_B_label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

    def backward(self):
        """Calculate the loss for generators G_A and G_B"""

        self.loss_C_A = 0
        self.loss_C_B = 0
        self.loss_C = 0
        if self.opt.netC == 'inception':
            self.logit, _ = self.netC(self.real)
        else:
            self.logit1 = self.netC(self.real2_A, self.real3_A)
            self.logit2 = self.netC(self.real2_B, self.real3_B)
            self.loss_C_A += 2 * self.criterionCE(self.logit1, target_label=self.train_A_label)
            self.loss_C_B += 2 * self.criterionCE(self.logit2, target_label=self.train_B_label)

        # self.loss_C = self.criterionCE(self.logit, self.train_label)
        self.loss_C = self.loss_C_A + self.loss_C_B
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

        size_A = np.shape(self.logit1)[0]
        size_B = np.shape(self.logit2)[0]
        total = size_A + size_B
        labels = torch.cat((self.train_A_label, self.train_B_label), 0).long()

        # for C_A : logit AB and logit BB
        outputs_A = torch.cat((self.logit1,self.logit2), 0)
        _, predicted = torch.max(outputs_A.data, 1)     # 每一行的最大值，即batch中每个样本的最大值
        correct = predicted.eq(labels.data).sum()
        self.train_C_acc = 100. * correct / total

        self.train_C_A_acc = self.train_C_acc
        self.train_C_B_acc = self.train_C_acc

    def test_acc(self):

        if self.isTrain:
            size_A = np.shape(self.test2_A)[0]
            size_B = np.shape(self.test2_B)[0]

            logit1 = self.netC(self.test2_A, self.test3_A)
            logit2 = self.netC(self.test2_B, self.test3_B)

            total = size_A + size_B
            labels = torch.cat((self.train_A_label, self.train_B_label), 0).long()
        else:
            size_A = np.shape(self.real2_A)[0]
            size_B = np.shape(self.real2_B)[0]

            logit1 = self.netC(self.real2_A, self.real3_A)
            logit2 = self.netC(self.real2_B, self.real3_B)

            total = size_A + size_B
            labels = torch.cat((self.test_A_label, self.test_B_label), 0).long()

        # for C_A : logit AB and logit BB
        outputs_A = torch.cat((logit1, logit2), 0)
        _, predicted_A = torch.max(outputs_A.data, 1)  # 每一行的最大值，即batch中每个样本的最大值
        correct_A = predicted_A.eq(labels.data).sum()
        self.test_C_acc = 100. * correct_A / total

        correct_B = correct_A
        correct = correct_A
        predicted_B = predicted_A
        predicted = predicted_A

        if self.opt.accs_or_logits == 'accs':
            return correct_A, correct_B, correct
        else:
            # return outputs1_A, outputs1_B, outputs1
            if self.opt.n_classes > 2:
                return predicted_A, predicted_B, predicted, labels.data
            else:
                return predicted_A, predicted_B, predicted
