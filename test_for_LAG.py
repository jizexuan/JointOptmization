import os
import numpy as np
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch
from util import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test(dataset_name, dataset_mode, k,K,epoch, suffix,test_for_traindata,max_dataset_size=-1):
    test_options = TestOptions()
    opt = test_options.parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    opt.dataroot = '../datasets/LAG_database_part_1/'
    opt.model = test_model
    opt.name = dataset_name + '_' + opt.model + suffix
    opt.dataset_mode = dataset_mode
    opt.netG = 'pac_unet_wavelet'
    # opt.netG = 'unet_128'
    opt.netD = 'basic'
    # opt.n_layers_D = 5
    opt.netC = 'inception'
    opt.direction = 'AtoB'
    opt.input_nc = 3
    opt.output_nc = 3
    opt.load_size = 299
    opt.crop_size = 299
    opt.preprocess = 'resize'
    opt.epoch = epoch
    opt.show_cam = False
    opt.accs_or_logits = 'labels'
    opt.test_for_traindata = test_for_traindata
    if max_dataset_size > 0:
        opt.max_dataset_size = max_dataset_size
    # opt.num_test = 1

    # fold
    opt.fold = K
    opt.cur_fold = k

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
    if hasattr(dataset.dataset,'A_size'):
        lenA = dataset.dataset.A_size
        lenB = dataset.dataset.B_size
    else:
        if test_for_traindata:
            lenA = dataset.dataset.train_A_size
            lenB = dataset.dataset.train_B_size
        else:
            lenA = dataset.dataset.test_A_size
            lenB = dataset.dataset.test_B_size
    len_all = lenA + lenB
    correctA0 = torch.Tensor([0])
    correctA1 = torch.Tensor([0])
    correctB0 = torch.Tensor([0])
    correctB1 = torch.Tensor([0])
    correct0 = torch.Tensor([0])
    correct1 = torch.Tensor([0])
    # labels = torch.Tensor([0,1])
    labels = np.array([0,1])

    cnt_A = 0
    cnt_B = 0
    for i, data in enumerate(dataset):
        # print(i)
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        # model.test()  # run inference
        # visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()  # get image paths
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        # print(model.eval())
        # print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        if opt.n_classes > 2:
            preA, preB, pre, labels = model.test_acc()
            # 2 classes: 0 [1 2]
            # preA[preA > 0] = 1
            # preB[preB > 0] = 1
            # labels[labels > 0] = 1
        else:
            preA, preB, pre = model.test_acc()

        if i < lenA:
            correctA0 += preA[0].eq(labels[0])
            correctB0 += preB[0].eq(labels[0])
            correct0 += pre[0].eq(labels[0])
            cnt_A += 1
        if i < lenB:
            correctA1 += preA[1].eq(labels[1])
            correctB1 += preB[1].eq(labels[1])
            correct1 += pre[1].eq(labels[1])
            cnt_B += 1

    webpage.save()  # save the HTML

    correctA = correctA0 + correctA1
    correctB = correctB0 + correctB1
    correct = correct0 + correct1

    accA = (1. * correctA.item() / len_all)
    accB = (1. * correctB.item() / len_all)
    acc = (1. * correct.item() / len_all)
    return accA, accB, acc

def writeTXT(path, name, data):
    np.savetxt("%s/%s" % (path, name), data, fmt="%f", delimiter="\t")

if __name__ == '__main__':

    test_model = 'resnet'
    dataset_name = 'LAG'
    dataset_mode = 'LAG'
    K = 5
    begin = 1
    end = 1
    # suffix = '_ResPacUnet_mx_skipGuide'
    suffix = '_inception'
    continue_test = False
    # test_for_traindata:
    test_for_traindata = False
    max_dataset_size = -1
    # max_dataset_size = 1711
    epoch_begin = 5
    epoch_total = 65
    epoch_count = int((epoch_total - epoch_begin) / 5) + 1
    for k in range(begin, end + 1, 1):

        ACCs_A = []
        ACCs_B = []
        ACCs = []
        # from 5 to 200
        epoch = range(epoch_begin, epoch_total + 1, 5)
        for i in epoch:
            epoch_str = str(i)
            path = './results/%s_%s_%d_%d%s/test_%s' % (dataset_name, test_model, k, K, suffix, epoch_str)
            if os.path.exists(path + '/ACCs_train') & test_for_traindata & continue_test:
                with open(path + '/ACCs_train', 'r') as file_to_read:
                    while True:
                        lines = file_to_read.readline()  # 整行读取数据
                        if not lines:
                            break
                            pass
                        epoch_tmp, ACC_A, ACC_B, ACC = [float(i) for i in lines.split('\t')]
            elif os.path.exists(path + '/ACCs_test') & (test_for_traindata == False) & continue_test:
                with open(path + '/ACCs_test', 'r') as file_to_read:
                    while True:
                        lines = file_to_read.readline()  # 整行读取数据
                        if not lines:
                            break
                            pass
                        epoch_tmp, ACC_A, ACC_B, ACC = [float(i) for i in lines.split('\t')]
            else:
                ACC_A, ACC_B, ACC = test(dataset_name,dataset_mode,k,K,epoch_str, '_%d_%d%s' % (k,K, suffix),test_for_traindata,max_dataset_size=max_dataset_size)
                data = np.array([np.float(i), ACC_A, ACC_B, ACC]).reshape(1, 4)
                if test_for_traindata:
                    writeTXT(path, 'ACCs_train', data)
                else:
                    writeTXT(path, 'ACCs_test', data)
            ACCs_A.append(ACC_A)
            ACCs_B.append(ACC_B)
            ACCs.append(ACC)

        # all the accs
        path = "./results/%s_%s_%d_%d%s" % (dataset_name, test_model, k, K, suffix)
        if not os.path.exists(path):
            os.makedirs(path)

        data = np.concatenate((np.linspace(epoch_begin, epoch_total, epoch_count).reshape(epoch_count, 1), np.array(ACCs_A).reshape(epoch_count, 1), np.array(ACCs_B).reshape(epoch_count, 1), np.array(ACCs).reshape(epoch_count, 1)), axis=1)
        if test_for_traindata:
            writeTXT(path, 'ACCs_train', data)
        else:
            writeTXT(path, 'ACCs_test', data)

        # plot
        plt.figure()
        x_axis_data = range(epoch_begin, epoch_total + 1, 5)
        plt_A, = plt.plot(x_axis_data, ACCs_A)
        plt_B, = plt.plot(x_axis_data, ACCs_B)
        plt_, = plt.plot(x_axis_data, ACCs)
        plt.legend(handles=[plt_A, plt_B, plt_], labels=['A', 'B', 'fusion'])
        plt.xlabel('epoch')
        plt.ylabel('ACCs')
        if test_for_traindata:
            plt.savefig("%s/ACCs_train.jpg" % (path))
        else:
            plt.savefig("%s/ACCs_test.jpg" % (path))
        # plt.show()
        # plt.close()
        print('ciao!')
