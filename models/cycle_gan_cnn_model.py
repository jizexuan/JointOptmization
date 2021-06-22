import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import cv2
from util import util
from torch.nn import functional as F
import gc
import math



class CycleGANCNNModel(BaseModel):
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
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_ce', type=float, default=1.0, help='weight for CE loss')
            parser.add_argument('--lambda_ce_A', type=float, default=1.0, help='weight for CE loss')
            parser.add_argument('--lambda_ce_B', type=float, default=1.0, help='weight for CE loss')
            parser.add_argument('--lambda_gp', type=float, default=0.0, help='weight for WGAN gradient penalty')
            parser.add_argument('--optimize_D', type=bool, default=True, help='fix D , means D dont optimize' )
            parser.add_argument('--add_c_after', type=int, default=0, help='C start from which epoch')
            parser.add_argument('--add_eig_loss', type=bool, default=False, help='whether use eig loss')
            parser.add_argument('--lambda_euc', type=float, default=0.0, help='whether use Euclidean loss')
        parser.add_argument('--n_classes', type=int, default=2, help='how many class')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_A', 'cycle_A', 'idt_A','D_A', 'G_B', 'cycle_B', 'idt_B','D_B']
        if self.opt.use_C:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'A_CE', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'B_CE']
            # self.loss_names.append('A_CE')
            # self.loss_names.append('B_CE')
            if self.opt.show_cam:
                self.opt.display_ncols = 9
            else:
                self.opt.display_ncols = 6
        else:
            self.opt.show_cam = False
            self.opt.show_acc = False
            self.opt.display_ncols = 4
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if (self.opt.dataset_mode == 'uwf') & (self.isTrain) :
            visual_names_A = ['real_A_rgb', 'fake_B_rgb', 'rec_A_rgb']
            visual_names_B = ['real_B_rgb', 'fake_A_rgb', 'rec_B_rgb']
        else:
            visual_names_A = ['real_A', 'fake_B', 'rec_A']
            visual_names_B = ['real_B', 'fake_A', 'rec_B']

        if self.opt.use_C:
            visual_names_A.append('diff_AB_heatmap')
            visual_names_A.append('diff_BB_heatmap')
            visual_names_B.append('diff_BA_heatmap')
            visual_names_B.append('diff_AA_heatmap')

        if self.opt.show_cam & self.opt.use_C:
            visual_names_A.append('cam_AB')
            visual_names_A.append('cam_BB')
            visual_names_A.append('cam_B')
            visual_names_B.append('cam_BA')
            visual_names_B.append('cam_AA')
            visual_names_B.append('cam_A')

        if self.opt.dataset_mode == 'uwf':
            visual_names_A.append('idt_B_rgb')
            visual_names_B.append('idt_A_rgb')
        else:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if not self.isTrain:
            visual_names_A.append('diff_AB')
            visual_names_B.append('diff_BA')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B','D_A','D_B']
            if self.opt.use_C:
                self.model_names.append('C_A')
                self.model_names.append('C_B')
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
            if self.opt.use_C:
                self.model_names.append('C_A')
                self.model_names.append('C_B')

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.opt.use_C:
            self.netC_A = networks.define_C(opt.output_nc, opt.ndf, opt.netC,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.n_classes)
            self.netC_B = networks.define_C(opt.output_nc, opt.ndf, opt.netC,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.n_classes)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            if opt.add_eig_loss:
                self.criterionEig = torch.nn.MSELoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.opt.optimize_D:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if self.opt.use_C:
                self.criterionCE = networks.CELoss(dataset_mode=opt.dataset_mode, n_classes=opt.n_classes, lambda_ce_A=self.opt.lambda_ce_A, lambda_ce_B=self.opt.lambda_ce_B, device=self.device).to(self.device)
                self.optimizer_GC = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netC_A.parameters(),self.netC_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_GC)
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

    def show_heatmap(self, tensor_img, tensor_mask):
        img = util.tensor2im(tensor_img)
        mask = util.tensor2im(tensor_mask)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return heatmap * 255

    def cam2img(self,cam):
        output_cam = []
        # generate the class activation maps upsample to 256x256
        size_upsample = (self.opt.crop_size, self.opt.crop_size)

        # cam = np.maximum(cam, 0)  # relu
        cam = cam - np.min(cam)
        cam_img = cam / (np.max(cam) - np.min(cam))
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
            cam = weight_softmax[idx].dot(feature_conv[0].reshape((nc, h * w)))     # this is CAM vector    dot() means matrix multi
            cam = cam.reshape(h, w)     # into a 2D array
        return cam

    def hook_feature(self, module, input, output):
        # self.features_blobs.append(output.data.cpu().numpy())
        self.features_blobs.append(output.data)

    def make_hook(self, isA):
        # hook the feature extractor
        finalconv_name = 'layer4'
        self.features_blobs = []
        if isA == True:
            self.netC_A._modules.get('module')._modules.get(finalconv_name).register_forward_hook(self.hook_feature)
            # self.netC_A._modules.get(finalconv_name).register_forward_hook(self.hook_feature)
        else:
            self.netC_B._modules.get('module')._modules.get(finalconv_name).register_forward_hook(self.hook_feature)


    def show_CAM(self, features_blobs, logit, isA):
        # get the softmax weight
        if isA == True:
            params = list(self.netC_A.parameters())
            # weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
            weight_softmax = np.squeeze(params[-2].data)
            # h_x = F.softmax(logit[0]).data.squeeze()
            # probs, idx = h_x.sort(0, True)
            # probs = probs.cpu().numpy()
            # idx = idx.cpu().numpy()
            # generate class activation mapping for the top1 prediction
            cam = self.returnCAM(features_blobs[0], weight_softmax, [0])
        else:
            params = list(self.netC_B.parameters())
            # weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
            weight_softmax = np.squeeze(params[-2].data)    # params[-2].shape [2,512]
            # h_x = F.softmax(logit[0]).data.squeeze()
            # probs, idx = h_x.sort(0, True)
            # probs = probs.cpu().numpy()
            # idx = idx.cpu().numpy()
            # generate class activation mapping for the top1 prediction
            cam = self.returnCAM(features_blobs[0], weight_softmax, [1])

        heatmap = self.cam2img(cam)

        img = torch.stack((self.real_A[0][0],self.real_A[0][0],self.real_A[0][0]),dim=2).cpu().numpy()
        # height, width, _ = img.shape
        # result = heatmap * 0.1 + img * 0.5

        # del logit, params, weight_softmax, h_x, CAMs, img, height, width, _ , heatmap

        return cam,heatmap

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        if self.opt.use_C:
            # norm_layer = networks.get_norm_layer(norm_type='instance')
            # self.diff_AB = norm_layer(self.real_A - self.fake_B)
            self.diff_AB = networks.instance_normalization(self.real_A - self.fake_B, self.opt.input_nc)
            self.diff_BA = networks.instance_normalization(self.real_B - self.fake_A, self.opt.input_nc)

            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

            self.diff_BB = networks.instance_normalization(self.real_B - self.idt_A, self.opt.input_nc)
            self.diff_AA = networks.instance_normalization(self.real_A - self.idt_B, self.opt.input_nc)

            if self.opt.dataset_mode == 'uwf':
                self.real_A_rgb = self.real_A[:,0:3,:,:]
                self.real_B_rgb = self.real_B[:,0:3,:,:]

                self.fake_B_rgb = self.fake_B[:,0:3,:,:]
                self.rec_A_rgb = self.rec_A[:,0:3,:,:]
                self.fake_A_rgb = self.fake_A[:,0:3,:,:]
                self.rec_B_rgb = self.rec_B[:,0:3,:,:]

                self.diff_AB_rgb = self.diff_AB[:,0:3,:,:]
                self.diff_BA_rgb = self.diff_BA[:,0:3,:,:]

                self.diff_BB_rgb = self.diff_BB[:, 0:3, :, :]
                self.diff_AA_rgb = self.diff_AA[:, 0:3, :, :]

                self.idt_A_rgb = self.idt_A[:, 0:3, :, :]
                self.idt_B_rgb = self.idt_B[:, 0:3, :, :]

                self.diff_AB_heatmap = self.show_heatmap(self.real_A_rgb, self.diff_AB_rgb)
                self.diff_BA_heatmap = self.show_heatmap(self.real_B_rgb, self.diff_BA_rgb)

                self.diff_BB_heatmap = self.show_heatmap(self.real_B_rgb, self.diff_BB_rgb)
                self.diff_AA_heatmap = self.show_heatmap(self.real_A_rgb, self.diff_AA_rgb)
            else:
                self.diff_AB_heatmap = self.show_heatmap(self.real_A, self.diff_AB)
                self.diff_BA_heatmap = self.show_heatmap(self.real_B, self.diff_BA)
                self.diff_BB_heatmap = self.show_heatmap(self.real_B, self.diff_BB)
                self.diff_AA_heatmap = self.show_heatmap(self.real_A, self.diff_AA)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D = loss_D_fake
        # WGAN GP
        if self.opt.lambda_gp > 0:
            loss_gp, gradients = networks.cal_gradient_penalty(netD,real,fake,torch.device('cuda:{}'.format(self.gpu_ids[0])),type='mixed', constant=1.0, lambda_gp=self.opt.lambda_gp)
            loss_D += loss_gp
        if self.opt.optimize_D:
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_GC(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # 这里 real_A --G_A--> fake_B --G_B--> rec_A
        # 其中 判别fake_B的判别器是D_A (与常识相反)
        # 同理 判别fake_A的判别器是D_B

        self.logit_AB = self.netC_A(self.diff_AB)
        self.logit_BB = self.netC_A(self.diff_BB)
        self.logit_BA = self.netC_B(self.diff_BA)
        self.logit_AA = self.netC_B(self.diff_AA)

        if self.opt.show_cam:
            self.make_hook(isA=True)
            self.cam_AB_ori, self.cam_AB = self.show_CAM(self.features_blobs, self.logit_AB, isA=True)
            self.loss_A_CE = self.criterionCE(self.logit_AB, target_label=self.train_A_label)  # target is A

            self.make_hook(isA=False)
            self.cam_BB_ori,self.cam_BB = self.show_CAM(self.features_blobs, self.logit_BB, isA=False)
            self.loss_A_CE += self.criterionCE(self.logit_BB, target_label=self.train_B_label)  # target is B

            self.make_hook(isA=False)
            self.cam_BA_ori, self.cam_BA = self.show_CAM(self.features_blobs, self.logit_BA, isA=False)
            self.loss_B_CE = self.criterionCE(self.logit_BA, target_label=self.train_B_label)  # target is B

            self.make_hook(isA=True)
            self.cam_AA_ori, self.cam_AA = self.show_CAM(self.features_blobs, self.logit_AA, isA=True)
            self.loss_B_CE += self.criterionCE(self.logit_AA, target_label=self.train_A_label)  # target is A

            self.cam_B = self.cam2img(self.cam_BA_ori + self.cam_BB_ori)
            self.cam_A = self.cam2img(self.cam_AB_ori + self.cam_AA_ori)

            # del self.features_blobs, logit
            # gc.collect()
        else:
            self.loss_A_CE = self.criterionCE(self.netC_A(self.diff_AB), target_label=self.train_A_label)  # target is A
            self.loss_A_CE += self.criterionCE(self.netC_A(self.diff_BB), target_label=self.train_B_label)  # target is B
            self.loss_B_CE = self.criterionCE(self.netC_B(self.diff_BA), target_label=self.train_B_label)  # target is B
            self.loss_B_CE += self.criterionCE(self.netC_B(self.diff_AA), target_label=self.train_A_label)  # target is A

        self.loss_A_CE *= self.opt.lambda_ce
        self.loss_B_CE *= self.opt.lambda_ce

        # combined loss and calculate gradients
        self.loss_GC = self.loss_A_CE + self.loss_B_CE + self.loss_idt_A + self.loss_idt_B
        self.loss_GC.backward(retain_graph=True)

    def eval_eig(self, real, fake):
        if real.shape[1] <= 3:
            if real.shape[1] == 3:
                real_gray = 0.299 * real[0,:,:] + 0.587 * real[1,:,:] + 0.114 * real[2,:,:]
                real_e, v = torch.symeig(real_gray, eigenvectors=True)
                fake_gray = 0.299 * fake[0, :, :] + 0.587 * fake[1, :, :] + 0.114 * fake[2, :, :]
                fake_e, v = torch.symeig(fake_gray, eigenvectors=True)
            else:
                real_e, v = torch.symeig(real[0,:,:], eigenvectors=True)
                fake_e, v = torch.symeig(fake[0, :, :], eigenvectors=True)
            loss = self.criterionEig(real_e, fake_e)
        else:
            loss = 0
            # color
            real_gray = 0.299 * real[0, :, :] + 0.587 * real[1, :, :] + 0.114 * real[2, :, :]
            real_e, v = torch.symeig(real_gray, eigenvectors=True)
            fake_gray = 0.299 * fake[0, :, :] + 0.587 * fake[1, :, :] + 0.114 * fake[2, :, :]
            fake_e, v = torch.symeig(fake_gray, eigenvectors=True)
            loss += self.criterionEig(real_e, fake_e)
            # gray1
            real_e, v = torch.symeig(real[3, :, :], eigenvectors=True)
            fake_e, v = torch.symeig(fake[3, :, :], eigenvectors=True)
            loss += self.criterionEig(real_e, fake_e)
            # gray2
            real_e, v = torch.symeig(real[4, :, :], eigenvectors=True)
            fake_e, v = torch.symeig(fake[4, :, :], eigenvectors=True)
            loss += self.criterionEig(real_e, fake_e)
            loss /= 3

        return loss

    def euclidean(self, x, y):
        sigma = 1
        t = 1.0 / (2.0 * math.pi * (sigma ^ 2))
        k = (x - y) ** 2
        r = - k / (2.0 * (sigma ^ 2))
        G = (t * torch.exp(r))
        D = (x - y)
        dist = (D * G).sum()
        return dist

    def eval_euc_dist(self, real, fake):
        if real.shape[0] <= 3:
            if real.shape[1] == 3:
                real_gray = 0.299 * real[0,:,:] + 0.587 * real[1,:,:] + 0.114 * real[2,:,:]
                fake_gray = 0.299 * fake[0, :, :] + 0.587 * fake[1, :, :] + 0.114 * fake[2, :, :]
                real_flatten = torch.flatten(real_gray)
                fake_flatten = torch.flatten(fake_gray)
            else:
                real_flatten = torch.flatten(real[0, :, :])
                fake_flatten = torch.flatten(fake[0, :, :])
            loss = self.euclidean(real_flatten, fake_flatten)
        else:
            loss = 0
            # color
            real_gray = 0.299 * real[0, :, :] + 0.587 * real[1, :, :] + 0.114 * real[2, :, :]
            fake_gray = 0.299 * fake[0, :, :] + 0.587 * fake[1, :, :] + 0.114 * fake[2, :, :]
            real_flatten = torch.flatten(real_gray)
            fake_flatten = torch.flatten(fake_gray)
            loss += self.euclidean(real_flatten, fake_flatten)
            # gray1
            real_flatten = torch.flatten(real[3, :, :])
            fake_flatten = torch.flatten(fake[3, :, :])
            loss += self.euclidean(real_flatten, fake_flatten)
            # gray2
            real_flatten = torch.flatten(real[4, :, :])
            fake_flatten = torch.flatten(fake[4, :, :])
            loss += self.euclidean(real_flatten, fake_flatten)
            loss /= 3

        return loss

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            if (not self.opt.use_C) | (self.opt.add_c_after > 0):
                self.idt_A = self.netG_A(self.real_B)

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            if (not self.opt.use_C) | (self.opt.add_c_after > 0):
                self.idt_B = self.netG_B(self.real_A)

            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # 这里 real_A --G_A--> fake_B --G_B--> rec_A
        # 其中 判别fake_B的判别器是D_A (与常识相反)
        # 同理 判别fake_A的判别器是D_B

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # eig loss
        if self.opt.add_eig_loss:
            self.loss_eig_A = 0
            self.loss_eig_B = 0
            for i in range(self.real_A.shape[0]):
                self.loss_eig_A += self.eval_eig(self.real_A[i,:,:,:], self.fake_B[i,:,:,:])
                self.loss_eig_B += self.eval_eig(self.real_B[i,:,:,:], self.fake_A[i,:,:,:])
            self.loss_G += self.loss_eig_A / self.opt.batch_size + self.loss_eig_B / self.opt.batch_size
        # euc loss
        if self.opt.lambda_euc != 0:
            self.loss_euc_A = 0
            self.loss_euc_B = 0
            for i in range(self.real_A.shape[0]):
                self.loss_euc_A += self.eval_euc_dist(self.real_A[i,:,:,:], self.fake_B[i,:,:,:])
                self.loss_euc_B += self.eval_euc_dist(self.real_B[i,:,:,:], self.fake_A[i,:,:,:])
            self.loss_G += (self.loss_euc_A / self.opt.batch_size + self.loss_euc_B / self.opt.batch_size) * self.opt.lambda_euc
        self.loss_G.backward()

    def optimize_parameters(self,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        if epoch >= self.opt.add_c_after:
            self.opt.use_C = True
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_GC.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_GC()  # calculate gradients for G_A and G_B
            self.optimizer_GC.step()  # update G_A and G_B's weights

            # self.forward()

        # G_A and G_B
        if self.opt.optimize_D:
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        if self.opt.optimize_D:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero

        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B

        if self.opt.optimize_D:
            self.optimizer_D.step()  # update D_A and D_B's weights

        # gc.collect()

    def train_acc(self):

        size_A = np.shape(self.logit_AB)[0]
        size_B = np.shape(self.logit_BB)[0]
        total = size_A + size_B
        labels = torch.cat((self.train_A_label, self.train_B_label), 0).long()

        # for C_A : logit AB and logit BB
        outputs_A = torch.cat((self.logit_AB,self.logit_BB), 0)
        _, predicted = torch.max(outputs_A.data, 1)     # 每一行的最大值，即batch中每个样本的最大值
        correct = predicted.eq(labels.data).sum()
        self.train_C_A_acc = 100. * correct / total

        # for C_B : logit AA and logit BA
        outputs_B = torch.cat((self.logit_AA, self.logit_BA), 0)
        _, predicted = torch.max(outputs_B.data, 1)  # 每一行的最大值，即batch中每个样本的最大值
        correct = predicted.eq(labels.data).sum()
        self.train_C_B_acc = 100. * correct / total

        # for C : C_A and C_B
        outputs = torch.cat((F.softmax(outputs_A,1), F.softmax(outputs_B,1)), 1)
        _, predicted = torch.max(F.softmax(outputs,1).data, 1)  # 每一行的最大值，即batch中每个样本的最大值
        predicted %= 2
        correct = predicted.eq(labels.data).sum()
        self.train_C_acc = 100. * correct / total

    def test_acc(self):

        if self.isTrain:
            size_A = np.shape(self.test_A)[0]
            size_B = np.shape(self.test_B)[0]
            fake_B = self.netG_A(self.test_A)
            idt_A = self.netG_A(self.test_B)
            diff_AB = networks.instance_normalization(self.test_A - fake_B, self.opt.input_nc)
            diff_BB = networks.instance_normalization(self.test_B - idt_A, self.opt.input_nc)

            fake_A = self.netG_B(self.test_B)
            idt_B = self.netG_B(self.test_A)
            diff_BA = networks.instance_normalization(self.test_B - fake_A, self.opt.input_nc)
            diff_AA = networks.instance_normalization(self.test_A - idt_B, self.opt.input_nc)

            total = size_A + size_B
            labels = torch.cat((self.train_A_label, self.train_B_label), 0).long()
        else:
            size_A = np.shape(self.real_A)[0]
            size_B = np.shape(self.real_B)[0]
            fake_B = self.netG_A(self.real_A)
            idt_A = self.netG_A(self.real_B)
            diff_AB = networks.instance_normalization(self.real_A - fake_B, self.opt.input_nc)
            diff_BB = networks.instance_normalization(self.real_B - idt_A, self.opt.input_nc)

            fake_A = self.netG_B(self.real_B)
            idt_B = self.netG_B(self.real_A)
            diff_BA = networks.instance_normalization(self.real_B - fake_A, self.opt.input_nc)
            diff_AA = networks.instance_normalization(self.real_A - idt_B, self.opt.input_nc)

            total = size_A + size_B
            # labels = torch.cat((torch.full([size_A], self.test_A_label), torch.full([size_B], self.test_B_label)), 0).long().cuda()
            labels = torch.cat((self.test_A_label, self.test_B_label), 0).long()

        logit_AB = self.netC_A(diff_AB)
        logit_BB = self.netC_A(diff_BB)
        logit_BA = self.netC_B(diff_BA)
        logit_AA = self.netC_B(diff_AA)

        # for C_A : logit AB and logit BB
        outputs_A = torch.cat((logit_AB,logit_BB),0)
        _, predicted_A = torch.max(outputs_A.data, 1)     # 每一行的最大值，即batch中每个样本的最大值
        correct_A = predicted_A.eq(labels.data).sum()
        self.test_C_A_acc = 100. * correct_A / total

        # for C_B : logit AA and logit BA
        outputs_B = torch.cat((logit_AA,logit_BA),0)
        _, predicted_B = torch.max(outputs_B.data, 1)  # 每一行的最大值，即batch中每个样本的最大值
        correct_B = predicted_B.eq(labels.data).sum()
        self.test_C_B_acc = 100. * correct_B / total

        # for C : C_A and C_B
        outputs = torch.cat((F.softmax(outputs_A,1), F.softmax(outputs_B,1)), 1)
        _, predicted = torch.max(F.softmax(outputs,1).data, 1)  # 每一行的最大值，即batch中每个样本的最大值
        predicted %= self.opt.n_classes
        correct = predicted.eq(labels.data).sum()
        self.test_C_acc = 100. * correct / total

        if self.opt.accs_or_logits == 'accs':
            return correct_A, correct_B, correct
        elif self.opt.accs_or_logits == 'labels':
            # return outputs_A, outputs_B, outputs
            if self.opt.n_classes > 2:
                return predicted_A, predicted_B, predicted, labels.data
            else:
                return predicted_A, predicted_B, predicted
        elif self.opt.accs_or_logits == 'logits':
            return outputs_A, outputs_B, outputs
