import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html, util
import matplotlib
import torch
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
matplotlib.use('Agg')

def roc(y,scores,model_name,dataset_name,pos_label=1):
    # y = np.array([1, 1, 2, 2])
    # scores = np.array([0.1, 0.4, 0.35, 0.8])

    # parameter = 40
    # y = np.random.randint(0, 2, size=parameter)
    # scores = np.random.choice(np.arange(0.1, 1, 0.1), parameter)

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # A
    fpr, tpr, thresholds = metrics.roc_curve(y, scores[:,0], pos_label=pos_label)
    auc_A = metrics.auc(fpr, tpr)
    plt_A, = plt.plot(fpr, tpr, label='ROC of A (AUC = %0.2f)' % auc_A)

    # B
    fpr, tpr, thresholds = metrics.roc_curve(y, scores[:, 1], pos_label=pos_label)
    auc_B = metrics.auc(fpr, tpr)
    plt_B, = plt.plot(fpr, tpr, label='ROC of B (AUC = %0.2f)' % auc_B)


    # fusion
    fpr, tpr, thresholds = metrics.roc_curve(y, scores[:, 2], pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    plt_, = plt.plot(fpr, tpr, label='ROC of fusion (AUC = %0.2f)' % auc)

    # save
    data = np.concatenate([fpr.reshape(fpr.size, 1), tpr.reshape(tpr.size, 1)], axis=1)
    util.mkdir("../result_txts/" + dataset_name)
    roc_data = Path("../result_txts/" + dataset_name + '/' + model_name)
    if roc_data.is_file():
        print('WARNING: %s is alreadly exist!' % roc_data)
        print('Rewriting ...')
        os.remove(roc_data)
    # else:
    #     np.savetxt(roc_data, data, fmt="%f", delimiter="\t")
    np.savetxt(roc_data, data, fmt="%f", delimiter="\t")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(handles=[plt_A, plt_B, plt_])
    plt.savefig('./results/%s/ROC_test.jpg' % model_name)
    return auc

def Precision_Recall_F1_score(y_true, y_pred):

    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1score = f1_score(y_true, y_pred, average='binary')

    print('precision: %f' % p)
    print('recall: %f' % r)
    print('F1 score: %f' % f1score)
    # print(classification_report(y_true, y_pred, target_names=['normal', 'abnormal']))

if __name__ == '__main__':

    K = 1
    begin = 1
    end = 1
    model = 'resnet'
    suffix = '_resnet50_torchvision'
    # suffix = ''
    epoch = '20'
    dataset_name = 'AMD'

    for k in range(begin, end + 1, 1):
        test_options = TestOptions()
        opt = test_options.parse()  # get test options
        # hard-code some parameters for test

        model_name = '%s_%s_%d_%d%s' % (dataset_name, model, k,K, suffix)
        opt.dataroot = '../datasets/IDRID/'
        opt.name = model_name
        opt.dataset_mode = dataset_name
        opt.model = model
        opt.num_threads = 1  # test code only supports num_threads = 1
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        # opt.gpu_ids = gpu_id
        # opt.num_test = 2000
        # opt.netG = 'unet_128'
        opt.netG = 'res_pac_unet'
        opt.netD = 'basic'
        opt.netC = 'resnet50_torchvision'
        opt.direction = 'AtoB'
        opt.input_nc = 3
        opt.output_nc = 3
        opt.load_size = 512
        opt.crop_size = 512
        opt.epoch = epoch

        # fold
        opt.fold = K
        opt.cur_fold = k

        opt.preprocess = 'resize'
        opt.show_cam = False
        opt.test_for_traindata = False
        opt.accs_or_logits = 'logits'
        test_options.print_options(opt)

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        if opt.eval:
            model.eval()
        if hasattr(dataset.dataset,'A_size'):
            lenA = dataset.dataset.A_size
            lenB = dataset.dataset.B_size
        else:
            if opt.test_for_traindata:
                lenA = dataset.dataset.train_A_size
                lenB = dataset.dataset.train_B_size
            else:
                lenA = dataset.dataset.test_A_size
                lenB = dataset.dataset.test_B_size
        len_all = lenA + lenB
        logitA0 = []
        logitB0 = []
        logit0 = []
        logitA1 = []
        logitB1 = []
        logit1 = []
        pre0 = []
        pre1 = []
        # labels = np.array([0,1])

        abnormal_index = 1

        cnt_A = 0
        cnt_B = 0
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            img_path = model.get_image_paths()  # get image paths
            # if i % 5 == 0:  # save images to an HTML file
            #     print('processing (%04d)-th image... %s' % (i, img_path))
            logitA, logitB, logit = model.test_acc()
            if i < lenA:
                logitA0.append(logitA[0][abnormal_index].item())
                logitB0.append(logitB[0][abnormal_index].item())
                logit0.append(logit[0][abnormal_index].item())
                pre_list = logit[0].cpu().tolist()
                predicted = pre_list.index(max(pre_list))
                predicted %= 2
                pre0.append(predicted)
                cnt_A += 1
            if i < lenB:
                logitA1.append(logitA[1][abnormal_index].item())
                logitB1.append(logitB[1][abnormal_index].item())
                logit1.append(logit[1][abnormal_index].item())
                pre_list = logit[1].cpu().tolist()
                predicted = pre_list.index(max(pre_list))
                predicted %= 2
                pre1.append(predicted)
                cnt_B += 1

        logitA = np.concatenate((np.array(logitA0), np.array(logitA1)),axis=0)
        logitB = np.concatenate((np.array(logitB0), np.array(logitB1)),axis=0)
        logit = np.concatenate((np.array(logit0), np.array(logit1)),axis=0)
        scores = np.concatenate((logitA.reshape((len_all,1)),logitB.reshape((len_all,1)),logit.reshape((len_all,1))),axis=1)
        y = np.concatenate((np.zeros(lenA),np.ones(lenB)),axis=0).reshape((len_all,1))
        auc = roc(y, scores, model_name, opt.dataset_mode, pos_label=abnormal_index)

        pre = np.concatenate((np.array(pre0), np.array(pre1)),axis=0)
        Precision_Recall_F1_score(y, pre)

        print('auc: %f' % auc)
    print('ciao!')
