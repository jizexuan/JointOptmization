import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from util.visualizer import save_PM_imgs
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    K = 1
    begin = 1
    end = 1
    model = 'cycle_gan'
    suffix = ''

    for k in range(begin, end + 1, 1):
        test_options = TestOptions()
        opt = test_options.parse()  # get test options
        # hard-code some parameters for test
        model_name = 'PM_%s_%d_%d%s' % (model, k,K, suffix)
        opt.dataroot = '../datasets/PM/PALM-Training400/'
        opt.name = model_name
        opt.dataset_mode = 'PM'
        opt.model = model
        opt.num_threads = 1  # test code only supports num_threads = 1
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        # opt.gpu_ids = gpu_id
        # opt.num_test = 20
        # opt.netG = 'pac_unet_wavelet'
        opt.netG = 'unet_128'
        opt.netD = 'basic'
        # opt.netD = 'n_layers'
        # opt.n_layers_D = 5
        opt.netC = 'cnn'
        opt.direction = 'AtoB'
        opt.input_nc = 3
        opt.output_nc = 3
        opt.load_size = 128
        opt.crop_size = 128
        opt.epoch = '200'

        # fold
        opt.fold = K
        opt.cur_fold = k

        opt.preprocess = 'resize'
        opt.show_cam = False
        opt.test_for_traindata = False
        test_options.print_options(opt)

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_PM_imgs(visuals, img_path, model_name, opt.epoch)


    # webpage.save()  # save the HTML
    print('ciao!')