"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train_for_oct2octa.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train_for_oct2octa.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':

    K = 1
    begin = 1
    end = 1
    # suffix = '_ResPacUnet_mx_skipGuide'
    suffix = ''
    for k in range(begin,end+1,1):
        train_options = TrainOptions()
        opt = train_options.parse()  # get training options
        opt.dataroot = '../datasets/PM/PALM-Training400/'
        opt.model = 'cgan'
        opt.name = 'PM_' + opt.model + '_%d_%d' % (k,K) + suffix
        opt.dataset_mode = 'PM'
        opt.netG = 'unet_128'
        opt.netD = 'basic'
        # opt.netD = 'n_layers'
        # opt.n_layers_D = 5
        opt.netC = 'resnet50'
        opt.lr = 0.00002
        opt.direction = 'BtoA'
        opt.input_nc = 3
        opt.output_nc = 3
        opt.batch_size = 256

        # opt.display_ncols = 9
        opt.optimize_D = True
        opt.use_C = True
        # opt.lambda_gp = 10
        # opt.gan_mode = 'wgangp'
        # opt.max_dataset_size = 10000
        # opt.niter = 25
        # opt.niter_decay = 25
        # opt.epoch = '20'
        # opt.epoch_count = 21
        # opt.init_gain = 0.01
        # opt.lambda_ce = 0.0001
        # opt.display_freq = 1
        # opt.verbose = False
        opt.show_cam = False
        opt.show_acc = False
        opt.save_latest_freq = 1000
        opt.augment = True
        opt.lambda_L1 = 100

        # fold
        opt.fold = K
        opt.cur_fold = k

        # opt.lambda_euc = 0.00001

        opt.pool_size = 50
        opt.no_dropout = True
        opt.load_size = 128
        opt.crop_size = 128
        opt.preprocess = 'resize'
        opt.display_winsize = 256
        # use pretrained networks
        # opt.pretrain = True
        # opt.continue_train = True
        if opt.pretrain:
            opt.continue_train = True
            opt.epoch = '200'
            opt.pretrain_path = './checkpoints/PM_cycle_gan_cnn_1_1_ours'

        opt.display_env = opt.name
        opt.display_id = 1
        train_options.print_options(opt)

        # dataset
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        # model
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        total_iters = 0                # the total number of training iterations

        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights

                if (total_iters % opt.display_freq == 0) & (opt.model != 'resnet'):   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    # get losses
                    losses = model.get_current_losses()
                    # print to console
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    # print to visdom
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    # get and print accs
                    if hasattr(opt,'show_acc'):
                        if opt.show_acc:
                            model.train_acc()
                            model.test_acc()
                            accs = model.get_current_accs()
                            if opt.display_id > 0:
                                visualizer.plot_current_accs(epoch, float(epoch_iter) / dataset_size, accs)


                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()                     # update learning rates at the end of every epoch.
    print('ciao!')
