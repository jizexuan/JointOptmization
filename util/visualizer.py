import numpy as np
import pdb
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import torchvision.transforms as transforms
import cv2
# from scipy.misc import imresize
import matplotlib.image as mpimg
from skimage import measure
from skimage import transform
from skimage import morphology
from networks.pac import nd2col
from evaluation.evaluate_oct2octa import compute_ssim
from PIL import Image

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def custom_save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    # by Gloria
    # image_path : trainA trainB testA testB
    # A
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if label in ['real_A','test_A', 'real_B', 'test_B']:
            im = util.tensor2im(im_data)
            if label == 'real_A':
                short_path = ntpath.basename(image_path[0])
                name = os.path.splitext(short_path)[0]
                image_name = '%s.png' % (name)
                dir = os.path.join(image_dir,'trainA')
                if not os.path.exists(dir):
                    os.makedirs(dir)
                save_path = os.path.join(dir, image_name)
            elif label == 'test_A' :
                short_path = ntpath.basename(image_path[2])
                name = os.path.splitext(short_path)[0]
                image_name = '%s.png' % (name)
                dir = os.path.join(image_dir, 'testA')
                if not os.path.exists(dir):
                    os.makedirs(dir)
                save_path = os.path.join(dir, image_name)
            elif label == 'real_B' :
                short_path = ntpath.basename(image_path[1])
                name = os.path.splitext(short_path)[0]
                image_name = '%s.png' % (name)
                dir = os.path.join(image_dir, 'trainB')
                if not os.path.exists(dir):
                    os.makedirs(dir)
                save_path = os.path.join(dir, image_name)
            elif label == 'test_B':
                short_path = ntpath.basename(image_path[3])
                name = os.path.splitext(short_path)[0]
                image_name = '%s.png' % (name)
                dir = os.path.join(image_dir, 'testB')
                if not os.path.exists(dir):
                    os.makedirs(dir)
                save_path = os.path.join(dir, image_name)
            else:
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    # short_path = ntpath.basename(image_path[0])
    # name = os.path.splitext(short_path)[0]
    #
    # webpage.add_header(name)
    # ims, txts, links = [], [], []
    #
    # for label, im_data in visuals.items():
    #     im = util.tensor2im(im_data)
    #     image_name = '%s_%s.png' % (name, label)
    #     save_path = os.path.join(image_dir, image_name)
    #     h, w, _ = im.shape
    #     if aspect_ratio > 1.0:
    #         im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
    #     if aspect_ratio < 1.0:
    #         im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
    #     util.save_image(im, save_path)
    #
    #     ims.append(image_name)
    #     txts.append(label)
    #     links.append(image_name)
    # webpage.add_images(ims, txts, links, width=width)

    # by Gloria
    # A
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if label in ['real_A','fake_B','rec_A','diff_A_heatmap','diff_B_heatmap']:
            im = util.tensor2im(im_data)
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

    # B
    short_path = ntpath.basename(image_path[1])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if label in ['real_B','fake_A','rec_B','diff_A_heatmap','diff_B_heatmap']:
            im = util.tensor2im(im_data)
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

def tobinary(real, fake, diff, ratio):
    if ratio >= 1:
        threshold = ratio
    else:
        threshold = np.max(np.max(diff)) * ratio
    # threshold = 1
    # background = 50

    if diff.max == 1:
        return diff
    ret = bin = np.copy(diff)
    bin[diff < threshold] = 0
    bin[diff >= threshold] = 255
    # real_gray = util.rgb2gray(real)
    # fake_gray = util.rgb2gray(fake)
    # bin[real_gray < background] = 0
    # bin[fake_gray < background] = 0

    # ret[:,:,0] = ret[:,:,1] = ret[:,:,2] = (bin[:,:,0] | bin[:,:,1]) | bin[:,:,2]
    ret = bin

    return ret

def cam2img_for_COVIDXray(cam):
    output_cam = []
    # generate the class activation maps upsample to 256x256
    size_downsample = (16,16)

    # cam = np.maximum(cam, 0)  # relu
    cam = cam - np.min(cam)
    cam_img = cam / (np.max(cam) - np.min(cam))
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_downsample)
    cam_img = 255 - cam_img
    output = cv2.resize(cam_img, (128,128))
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    return output

def cam2img(cam):
    output_cam = []
    # generate the class activation maps upsample to 256x256
    h = cam.shape[0]
    w = cam.shape[1]
    # size_downsample = (8,8)
    # size_downsample = (np.int(h/16),np.int(w/16))
    size_downsample = (np.int(h/8),np.int(w/8))

    # cam = np.maximum(cam, 0)  # relu
    cam = cam - np.min(cam)
    cam_img = cam / (np.max(cam) - np.min(cam))
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_downsample)
    cam_img = 255 - cam_img
    output = cv2.resize(cam_img, (h, w))
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    return output

#将灰度数组映射为直方图字典,nums表示灰度的数量级
def arrayToHist(grayArray,nums):
    # if(len(grayArray.shape) != 2):
    #     print("length error")
    #     return None
    w,h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if(hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    #normalize
    n = w*h
    for key in hist.keys():
        hist[key] = float(hist[key])/n
    return hist

#直方图匹配函数，接受原始图像和目标灰度直方图
def histMatch(grayArray,h_d):
    #计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray,256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    #计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des

def fine_seg(diff,path):
    mask = mpimg.imread(path).copy()
    mask = cv2.resize(mask, (128, 128))
    mask[mask > 0] = 1
    mask[mask <= 0] = 0
    # mask = mask > 0
    mask = np.uint8(np.concatenate((np.expand_dims(mask,axis=2),np.expand_dims(mask,axis=2),np.expand_dims(mask,axis=2)), axis=2))
    ret = diff * mask
    return ret

def read_gray(path):
    ret = mpimg.imread(path)
    # ret = np.resize(ret,(128,128))
    ret = transform.resize(ret, (128, 128)) * 255
    ret = np.uint8(np.concatenate((np.expand_dims(ret, axis=2), np.expand_dims(ret, axis=2), np.expand_dims(ret, axis=2)), axis=2))
    # ret = np.uint8(ret)
    return ret

def add_diff2img(diff,real,color):
    # d = diff
    # d[:,:,1] = 0
    # d[:,:,2] = 0
    # ret = cv2.addWeighted(real, 1, d, 1, 0)
    h = real.shape[0]
    w = real.shape[1]

    d = np.zeros(diff.shape)
    # d = diff
    for i in range(h):
        for j in range(w):
            if diff[i,j,0] != 0:
                if color == 'red':
                    # red
                    d[i,j,0] = 255
                    d[i,j,1] = 99
                    d[i,j,2] = 71
                elif color == 'white':
                    # white
                    d[i, j, 0] = 255
                    d[i, j, 1] = 255
                    d[i, j, 2] = 255
                elif color == 'blue':
                    # blue
                    d[i, j, 0] = 65
                    d[i, j, 1] = 105
                    d[i, j, 2] = 255
            else:
                d[i, j, 0] = real[i, j, 0]
                d[i, j, 1] = real[i, j, 1]
                d[i, j, 2] = real[i, j, 2]

    return np.uint8(d)

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def get_5_region(img):
    mask = np.zeros(img.shape)
    labeled_img, num = measure.label(img, neighbors=4, background=0, return_num=True)
    props = measure.regionprops(labeled_img)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    indexes = np.argsort(numPix)
    for i in range(np.min([5,indexes.size])):
        index = indexes[indexes.size - 1 - i]
        mask[labeled_img == props[index].label] = 1
    return img * mask

def save_thumbnail_imgs_for_COVIDXray(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    # threshold = 70
    # threshold = 20

    # B
    short_path = ntpath.basename(image_path[1])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathB = os.path.join(image_dir, 'B')
    util.mkdirs(save_pathB)
    save_path = os.path.join(save_pathB, image_name)
    real_B = util.tensor2im(visuals['real_B'])
    fake_A = util.tensor2im(visuals['fake_A'])
    # hist_m = arrayToHist(real_B[:,:,0], 256)
    # fake_A = histMatch(fake_A[:,:,0],hist_m)
    # fake_A = np.uint8(np.concatenate((np.expand_dims(fake_A,axis=2),np.expand_dims(fake_A,axis=2),np.expand_dims(fake_A,axis=2)), axis=2))

    diff = np.int8(real_B) - np.int8(fake_A)
    diff = np.abs(diff)
    diff = util.rgb2gray(diff)
    threshold = np.max(np.max(diff)) * 0.1
    diff[diff < threshold] = 0
    diff[diff >= threshold] = 255
    # diff = np.uint8(tobinary(real_B,fake_A, diff))
    diff = fine_seg(diff,os.path.join('/data/zyni/zzy/pytorch-CycleGAN-and-pix2pix-master/datasets/B_xray_masked',short_path))
    cam = cam2img_for_COVIDXray(diff)
    # diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255
    # diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff,real_B,'red')
    stack = cv2.addWeighted(real_B,1,cam,0.5,0)
    img_set = np.concatenate((real_B, fake_A, diff_on_real, stack), axis=1)
    util.save_image(img_set, save_path)

def save_thumbnail_imgs(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    # threshold = 70
    # threshold = 20

    # B
    short_path = ntpath.basename(image_path[1])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathB = os.path.join(image_dir, 'B')
    util.mkdirs(save_pathB)
    save_path = os.path.join(save_pathB, image_name)
    real_B = util.tensor2im(visuals['real_B'])
    fake_A = util.tensor2im(visuals['fake_A'])
    # hist_m = arrayToHist(real_B[:,:,0], 256)
    # fake_A = histMatch(fake_A[:,:,0],hist_m)
    # fake_A = np.uint8(np.concatenate((np.expand_dims(fake_A,axis=2),np.expand_dims(fake_A,axis=2),np.expand_dims(fake_A,axis=2)), axis=2))

    diff = np.int8(real_B) - np.int8(fake_A)
    diff = np.abs(diff)
    diff = util.rgb2gray(diff)
    diff = np.uint8(tobinary(real_B,fake_A, diff))
    # diff = fine_seg(diff,os.path.join(image_dir, 'B_xray_masked',image_name))
    attention = read_gray('/data/zyni/zzy/datasets/LAG_database_part_1/splited_all/attention_map/testB/%s.jpg' % image_name[0:4])
    # diff = get_5_region(attention, diff)
    diff = cam2img(diff)
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff,real_B,'red')
    stack = cv2.addWeighted(real_B,1,diff,0.5,0)
    img_set = np.concatenate((real_B, fake_A, attention, diff_on_real, stack), axis=1)
    util.save_image(img_set, save_path)

def save_attention_map(visuals, image_path, model_name):
    image_dir = os.path.join('./results', model_name, 'test_latest','attention map')
    # A
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathA = os.path.join(image_dir, 'A')
    util.mkdirs(save_pathA)
    save_path = os.path.join(save_pathA, image_name)
    if 'diff_AB' in visuals:
        diff = util.tensor2im(visuals['diff_AB'])
    else:
        diff = util.tensor2im(visuals['diff_AA'])
    gray = util.rgb2gray(diff)
    util.save_image(gray, save_path)

    # B
    short_path = ntpath.basename(image_path[1])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathB = os.path.join(image_dir, 'B')
    util.mkdirs(save_pathB)
    save_path = os.path.join(save_pathB, image_name)
    real_B = util.tensor2im(visuals['real_B'])
    fake_A = util.tensor2im(visuals['fake_A'])
    # diff_BA = util.tensor2im(visuals['diff_BA'])
    # diff_BA = util.rgb2gray(diff_BA)
    diff = np.int8(real_B) - np.int8(fake_A)
    diff = np.abs(diff)
    diff = util.rgb2gray(diff)
    diff = np.uint8(tobinary(real_B, fake_A, diff))
    diff = cam2img(diff)
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255
    diff = np.uint8(diff)
    img_set = np.concatenate((real_B, diff), axis=1)
    util.save_image(img_set, save_path)

def save_diff_imgs(visuals, image_path, model_name):
    image_dir = os.path.join('./results', model_name, 'test_latest','diff')
    # B
    short_path = ntpath.basename(image_path[1])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathB = os.path.join(image_dir, 'B')
    util.mkdirs(save_pathB)
    save_path = os.path.join(save_pathB, image_name)
    real_B = util.tensor2im(visuals['real_B'])
    fake_A = util.tensor2im(visuals['fake_A'])
    # diff_BA = util.tensor2im(visuals['diff_BA'])
    # diff_BA = util.rgb2gray(diff_BA)
    diff = np.int8(real_B) - np.int8(fake_A)
    diff = np.abs(diff)
    diff = util.rgb2gray(diff)
    img_set = np.concatenate((real_B, diff), axis=1)
    util.save_image(img_set, save_path)

def save_fake_imgs(visuals, image_path, model_name):
    image_dir = os.path.join('./results', model_name, 'test_latest','fake')
    # A
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathA = os.path.join(image_dir, 'fake_B')
    util.mkdirs(save_pathA)
    save_path = os.path.join(save_pathA, image_name)
    fakeB = util.tensor2im(visuals['fake_B'])
    util.save_image(fakeB, save_path)

    # B
    short_path = ntpath.basename(image_path[1])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathB = os.path.join(image_dir, 'fake_A')
    util.mkdirs(save_pathB)
    save_path = os.path.join(save_pathB, image_name)
    fakeB = util.tensor2im(visuals['fake_A'])
    util.save_image(fakeB, save_path)

def save_uwf_imgs(visuals, image_path, model_name, epoch):
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'fakeA')
    # B
    split_path = image_path[1].split('/')
    name = split_path[4] + '_' + split_path[5] if len(split_path) == 6 else split_path[4]
    image_name = '%s.png' % (name)
    util.mkdirs(image_dir)
    real_B = util.tensor2im(visuals['real_B'])
    fake_A = util.tensor2im(visuals['fake_A'])
    diff_BA = np.int8(real_B) - np.int8(fake_A)
    diff_BA = np.abs(diff_BA)
    threshold_ratio = 0.2
    # rgb
    real = real_B[:,:,0:3]
    diff = util.rgb2gray(diff_BA[:,:,0:3])
    diff = np.uint8(tobinary(real, fake_A[:,:,0:3], diff, threshold_ratio))
    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'red')
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)
    rgb_set = np.concatenate((real, fake_A[:,:,0:3], diff_on_real, stack), axis=1)
    # early
    real = real_B[:, :, 3]
    diff = diff_BA[:, :, 3]
    fake = fake_A[:, :, 3]
    real = np.stack((real, real, real), axis=2)
    diff = np.stack((diff, diff, diff), axis=2)
    fake = np.stack((fake, fake, fake), axis=2)
    diff = np.uint8(tobinary(real, fake, diff, threshold_ratio))
    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'red')
    real = np.uint8(real)
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)
    early_set = np.concatenate((real, fake, diff_on_real, stack), axis=1)
    # late
    real = real_B[:, :, 4]
    diff = diff_BA[:, :, 4]
    fake = fake_A[:, :, 4]
    real = np.stack((real, real, real), axis=2)
    diff = np.stack((diff, diff, diff), axis=2)
    fake = np.stack((fake, fake, fake), axis=2)
    diff = np.uint8(tobinary(real, fake, diff, threshold_ratio))
    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'red')
    real = np.uint8(real)
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)
    late_set = np.concatenate((real, fake, diff_on_real, stack), axis=1)
    # save
    img_set = np.concatenate((rgb_set, early_set, late_set), axis=0)
    img_set = np.uint8(img_set)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(img_set, save_path)

def suppress_white_areas(img):
    threshold = np.max(img) * 0.5
    mask = np.copy(img)
    mask[img < threshold] = 0
    mask[img >= threshold] = 255
    labeled_img, num = measure.label(mask, neighbors=4, background=0, return_num=True)
    props = measure.regionprops(labeled_img)
    mean = np.mean(img)
    for ia in range(len(props)):
        if (props[ia].area >= 1000) & (props[ia].area < 5000):
            img[labeled_img == props[ia].label] = mean
    return img

def generate_diff_map(real, fake):
    real_cols = nd2col(real, kernel_size=5, stride=1, padding=2).cpu()
    fake_cols = nd2col(fake, kernel_size=5, stride=1, padding=2).cpu()
    diff = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            diff[i,j] = compute_ssim(real_cols[:,:,:,:,i,j].squeeze(), fake_cols[:,:,:,:,i,j].squeeze())
    diff = 1 - diff
    # diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255
    diff *= 255
    return diff

def save_uwf_imgs_for_multiCycleGAN_ssim(visuals, image_path, model_name, epoch):
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'fakeA_ssim_train')
    # B
    split_path = image_path[1].split('/')
    name = split_path[4] + '_' + split_path[5] if len(split_path) == 6 else split_path[4]
    image_name = '%s_%s.png' % (split_path[3], name)
    util.mkdirs(image_dir)
    thres = 0.5

    # early
    real2_B = util.tensor2im(visuals['real2_B'])[:,:,0]
    fake2_A = util.tensor2im(visuals['fake2_A'])[:,:,0]
    diff = generate_diff_map(visuals['real2_B'], visuals['fake2_A'])
    real = real2_B
    fake = fake2_A
    real = np.stack((real, real, real), axis=2)
    diff = np.stack((diff, diff, diff), axis=2)
    fake = np.stack((fake, fake, fake), axis=2)
    diff[diff >= thres] = 255
    diff[diff < thres] = 0
    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'red')
    real = np.uint8(real)
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)
    early_set = np.concatenate((real, fake, diff_on_real, stack), axis=1)

    # late
    real3_B = util.tensor2im(visuals['real3_B'])[:,:,0]
    fake3_A = util.tensor2im(visuals['fake3_A'])[:,:,0]
    diff = generate_diff_map(visuals['real3_B'], visuals['fake3_A'])
    real = real3_B
    fake = fake3_A
    real = np.stack((real, real, real), axis=2)
    diff = np.stack((diff, diff, diff), axis=2)
    fake = np.stack((fake, fake, fake), axis=2)
    diff[diff >= thres] = 255
    diff[diff < thres] = 0
    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real, 'red')
    real = np.uint8(real)
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)
    late_set = np.concatenate((real, fake, diff_on_real, stack), axis=1)

    # sav.name
    img_set = np.concatenate((early_set, late_set), axis=0)
    img_set = np.uint8(img_set)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(img_set, save_path)

def save_uwf_imgs_for_multiCycleGAN(visuals, image_path, model_name, epoch):
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'fakeA')
    only_early_late = 'onlyEarlyLate' in model_name
    # B
    split_path = image_path[1].split('/')
    name = split_path[4] + '_' + split_path[5] if len(split_path) == 6 else split_path[4]
    image_name = '%s_%s.png' % (split_path[3], name)
    util.mkdirs(image_dir)
    threshold_ratio = 0.2
    # rgb
    if not only_early_late:
        real1_B = util.tensor2im(visuals['real1_B'])
        fake1_A = util.tensor2im(visuals['fake1_A'])
        diff1_BA = np.int8(real1_B) - np.int8(fake1_A)
        diff1_BA = np.abs(diff1_BA)
        real = real1_B
        diff = util.rgb2gray(diff1_BA)
        diff = np.uint8(tobinary(real, fake1_A, diff, threshold_ratio))
        heatmap = cam2img(diff)
        diff = np.uint8(diff)
        diff_on_real = add_diff2img(diff, real,'red')
        stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)
        rgb_set = np.concatenate((real, fake1_A, diff_on_real, stack), axis=1)
    # early
    real2_B = util.tensor2im(visuals['real2_B'])[:,:,0]
    fake2_A = util.tensor2im(visuals['fake2_A'])[:,:,0]
    diff2_BA = np.int8(real2_B) - np.int8(fake2_A)
    diff = np.abs(diff2_BA)
    real = real2_B
    # real = suppress_white_areas(real)
    fake = fake2_A
    real = np.stack((real, real, real), axis=2)
    diff = np.stack((diff, diff, diff), axis=2)
    fake = np.stack((fake, fake, fake), axis=2)
    diff = np.uint8(tobinary(real, fake, diff, threshold_ratio))
    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'red')
    real = np.uint8(real)
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)
    early_set = np.concatenate((real, fake, diff_on_real, stack), axis=1)
    # late
    real3_B = util.tensor2im(visuals['real3_B'])[:,:,0]
    fake3_A = util.tensor2im(visuals['fake3_A'])[:,:,0]
    diff3_BA = np.int8(real3_B) - np.int8(fake3_A)
    diff = np.abs(diff3_BA)
    real = real3_B
    # real = suppress_white_areas(real)
    fake = fake3_A
    real = np.stack((real, real, real), axis=2)
    diff = np.stack((diff, diff, diff), axis=2)
    fake = np.stack((fake, fake, fake), axis=2)
    diff = np.uint8(tobinary(real, fake, diff, threshold_ratio))
    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'red')
    real = np.uint8(real)
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)
    late_set = np.concatenate((real, fake, diff_on_real, stack), axis=1)
    # sav.name
    if only_early_late:
        img_set = np.concatenate((early_set, late_set), axis=0)
    else:
        img_set = np.concatenate((rgb_set, early_set, late_set), axis=0)
    img_set = np.uint8(img_set)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(img_set, save_path)

def save_uwf_imgs_for_multiCycleGAN_writeRealAndFake(i, visuals, image_path, model_name, epoch, fold):
    folds_idx = {
        1: {'normal': 1, 'npdr': 1, 'pdr': 1},
        2: {'normal': 25, 'npdr': 35, 'pdr': 23},
        3: {'normal': 49, 'npdr': 69, 'pdr': 44},
        4: {'normal': 73, 'npdr': 106, 'pdr': 66},
        5: {'normal': 97, 'npdr': 140, 'pdr': 89},
        6: {'normal': 120, 'npdr': 172, 'pdr': 110}
    }
    phase = 'test'
    abnormal_i = i + folds_idx[fold]['npdr'] - 1
    normal_i = i + folds_idx[fold]['normal'] - 1
    image_nameA = str(normal_i) + '.png'
    image_nameB = str(abnormal_i) + '.png'
    is_npdr = True
    if abnormal_i >= folds_idx[fold + 1]['npdr'] - 1:
        is_npdr = False
        image_nameB = str(abnormal_i - folds_idx[fold + 1]['npdr'] + folds_idx[fold]['pdr']) + '.png'
    normal_done = False
    if normal_i >= folds_idx[fold + 1]['normal'] - 1:
        normal_done = True

    # AtoB
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'diff_map_AtoB')
    # early
    if not normal_done:
        real2_A = util.tensor2im(visuals['real2_A'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'early', 'real')
        util.mkdirs(save_path)
        util.save_image(real2_A, os.path.join(save_path, image_nameA))
        fake2_B = util.tensor2im(visuals['fake2_B'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'early', 'fake')
        util.mkdirs(save_path)
        util.save_image(fake2_B, os.path.join(save_path, image_nameA))

    real2_B = util.tensor2im(visuals['real2_B'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'early', 'real')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'early', 'real')
    util.mkdirs(save_path)
    util.save_image(real2_B, os.path.join(save_path, image_nameB))
    idt2_A = util.tensor2im(visuals['idt2_A'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'early', 'fake')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'early', 'fake')
    util.mkdirs(save_path)
    util.save_image(idt2_A, os.path.join(save_path, image_nameB))
    # late
    if not normal_done:
        real3_A = util.tensor2im(visuals['real3_A'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'late', 'real')
        util.mkdirs(save_path)
        util.save_image(real3_A, os.path.join(save_path, image_nameA))
        fake3_B = util.tensor2im(visuals['fake3_B'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'late', 'fake')
        util.mkdirs(save_path)
        util.save_image(fake3_B, os.path.join(save_path, image_nameA))

    real3_B = util.tensor2im(visuals['real3_B'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'late', 'real')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'late', 'real')
    util.mkdirs(save_path)
    util.save_image(real3_B, os.path.join(save_path, image_nameB))
    idt3_A = util.tensor2im(visuals['idt3_A'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'late', 'fake')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'late', 'fake')
    util.mkdirs(save_path)
    util.save_image(idt3_A, os.path.join(save_path, image_nameB))

    # BtoA
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'diff_map_BtoA')
    # early
    real2_B = util.tensor2im(visuals['real2_B'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'early', 'real')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'early', 'real')
    util.mkdirs(save_path)
    util.save_image(real2_B, os.path.join(save_path, image_nameB))
    fake2_A = util.tensor2im(visuals['fake2_A'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'early', 'fake')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'early', 'fake')
    util.mkdirs(save_path)
    util.save_image(fake2_A, os.path.join(save_path, image_nameB))

    if not normal_done:
        real2_A = util.tensor2im(visuals['real2_A'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'early', 'real')
        util.mkdirs(save_path)
        util.save_image(real2_A, os.path.join(save_path, image_nameA))
        idt2_B = util.tensor2im(visuals['idt2_B'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'early', 'fake')
        util.mkdirs(save_path)
        util.save_image(idt2_B, os.path.join(save_path, image_nameA))
    # late
    real3_B = util.tensor2im(visuals['real3_B'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'late', 'real')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'late', 'real')
    util.mkdirs(save_path)
    util.save_image(real3_B, os.path.join(save_path, image_nameB))
    fake3_A = util.tensor2im(visuals['fake3_A'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'late', 'fake')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'late', 'fake')
    util.mkdirs(save_path)
    util.save_image(fake3_A, os.path.join(save_path, image_nameB))

    if not normal_done:
        real3_A = util.tensor2im(visuals['real3_A'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'late', 'real')
        util.mkdirs(save_path)
        util.save_image(real3_A, os.path.join(save_path, image_nameA))
        idt3_B = util.tensor2im(visuals['idt3_B'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'late', 'fake')
        util.mkdirs(save_path)
        util.save_image(idt3_B, os.path.join(save_path, image_nameA))

def save_uwf_imgs_for_multiCycleGAN_writeRealAndFake_onlyLate(i, visuals, image_path, model_name, epoch, fold):
    folds_idx = {
        1: {'normal': 1, 'npdr': 1, 'pdr': 1},
        2: {'normal': 25, 'npdr': 35, 'pdr': 23},
        3: {'normal': 49, 'npdr': 69, 'pdr': 44},
        4: {'normal': 73, 'npdr': 106, 'pdr': 66},
        5: {'normal': 97, 'npdr': 140, 'pdr': 89},
        6: {'normal': 120, 'npdr': 172, 'pdr': 110}
    }
    phase = 'test'
    abnormal_i = i + folds_idx[fold]['npdr'] - 1
    normal_i = i + folds_idx[fold]['normal'] - 1
    image_nameA = str(normal_i) + '.png'
    image_nameB = str(abnormal_i) + '.png'
    is_npdr = True
    if abnormal_i >= folds_idx[fold + 1]['npdr'] - 1:
        is_npdr = False
        image_nameB = str(abnormal_i - folds_idx[fold + 1]['npdr'] + folds_idx[fold]['pdr']) + '.png'
    normal_done = False
    if normal_i >= folds_idx[fold + 1]['normal'] - 1:
        normal_done = True

    # AtoB
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'diff_map_AtoB')
    # late
    if not normal_done:
        real3_A = util.tensor2im(visuals['real_A'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'late', 'real')
        util.mkdirs(save_path)
        util.save_image(real3_A, os.path.join(save_path, image_nameA))
        fake3_B = util.tensor2im(visuals['fake_B'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'late', 'fake')
        util.mkdirs(save_path)
        util.save_image(fake3_B, os.path.join(save_path, image_nameA))

    real3_B = util.tensor2im(visuals['real_B'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'late', 'real')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'late', 'real')
    util.mkdirs(save_path)
    util.save_image(real3_B, os.path.join(save_path, image_nameB))
    idt3_A = util.tensor2im(visuals['idt_A'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'late', 'fake')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'late', 'fake')
    util.mkdirs(save_path)
    util.save_image(idt3_A, os.path.join(save_path, image_nameB))

    # BtoA
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'diff_map_BtoA')
    # late
    real3_B = util.tensor2im(visuals['real_B'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'late', 'real')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'late', 'real')
    util.mkdirs(save_path)
    util.save_image(real3_B, os.path.join(save_path, image_nameB))
    fake3_A = util.tensor2im(visuals['fake_A'])[:,:,0]
    if is_npdr:
        save_path = os.path.join(image_dir, phase + '_npdr', 'late', 'fake')
    else:
        save_path = os.path.join(image_dir, phase + '_pdr', 'late', 'fake')
    util.mkdirs(save_path)
    util.save_image(fake3_A, os.path.join(save_path, image_nameB))

    if not normal_done:
        real3_A = util.tensor2im(visuals['real_A'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'late', 'real')
        util.mkdirs(save_path)
        util.save_image(real3_A, os.path.join(save_path, image_nameA))
        idt3_B = util.tensor2im(visuals['idt_B'])[:,:,0]
        save_path = os.path.join(image_dir, phase + '_normal', 'late', 'fake')
        util.mkdirs(save_path)
        util.save_image(idt3_B, os.path.join(save_path, image_nameA))

def save_diff_imgs(visuals, image_path, model_name, epoch):
    phase = 'test'
    # GA
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'diff_GA', phase + 'A')
    util.mkdirs(image_dir)
    split_path = image_path[1].split('/')
    name = split_path[4] + '_' + split_path[5] if len(split_path) == 6 else split_path[4]
    image_name = '%s.png' % (name)
    real_A = util.tensor2im(visuals['real1_B'])
    fake_A = util.tensor2im(visuals['fake1_A'])
    diff = np.int8(real_B) - np.int8(fake_A)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(diff, save_path)
    # GB
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'diff_GB', phase + 'A')
    util.mkdirs(image_dir)
    split_path = image_path[1].split('/')
    name = split_path[4] + '_' + split_path[5] if len(split_path) == 6 else split_path[4]
    image_name = '%s.png' % (name)
    real_B = util.tensor2im(visuals['real1_B'])
    fake_A = util.tensor2im(visuals['fake1_A'])
    diff = np.int8(real_B) - np.int8(fake_A)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(diff, save_path)

def save_LAG_imgs(visuals, image_path, model_name, epoch):
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'fakeA')
    # B
    short_path = ntpath.basename(image_path[1])
    # short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    util.mkdirs(image_dir)
    real_B = util.tensor2im(visuals['real_B'])
    fake_A = util.tensor2im(visuals['fake_A'])
    # real_B = util.tensor2im(visuals['real_A'])
    # fake_A = util.tensor2im(visuals['fake_B'])
    diff_BA = np.int8(real_B) - np.int8(fake_A)
    diff_BA = np.abs(diff_BA)
    # threshold_ratio = 0.3
    threshold_ratio = 0.5
    # rgb
    real = real_B[:, :, 0:3]
    diff = util.rgb2gray(diff_BA[:, :, 0:3])
    diff = np.uint8(tobinary(real, fake_A[:, :, 0:3], diff, threshold_ratio))
    # kernel = np.ones((3, 3), np.uint8)
    # diff = cv2.dilate(diff, kernel)
    # diff = cv2.erode(diff, kernel)
    # diff = morphology.remove_small_objects(diff, 2500, connectivity=2)
    arr = diff > 0
    cleaned = morphology.remove_small_objects(arr, min_size=100)
    cleaned = morphology.remove_small_holes(cleaned)
    diff = diff * cleaned

    # remove outer circle
    mask_circle = np.zeros_like(diff)
    w, h, ch = mask_circle.shape
    r = int(h / 2)
    cv2.circle(mask_circle, (r, r), r, (1, 1, 1), thickness=-1)
    diff = diff * mask_circle

    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'blue')
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)

    # read attention map
    attention_map = Image.open(os.path.join('../datasets/LAG_database_part_1/suspicious_glaucoma/attention_map', name + '.jpg')).convert('RGB')
    attention_map = attention_map.resize((w,h))
    attention_map = np.array(attention_map)

    img_set = np.concatenate((real, fake_A[:, :, 0:3], diff, stack, attention_map), axis=1)
    # save
    img_set = np.uint8(img_set)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(img_set, save_path)

def save_PM_imgs(visuals, image_path, model_name, epoch):
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'fakeA')
    # B
    short_path = ntpath.basename(image_path[1])
    # short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    util.mkdirs(image_dir)
    real_B = util.tensor2im(visuals['real_B'])
    fake_A = util.tensor2im(visuals['fake_A'])
    # real_B = util.tensor2im(visuals['real_A'])
    # fake_A = util.tensor2im(visuals['fake_B'])
    diff_BA = np.int8(real_B) - np.int8(fake_A)
    diff_BA = np.abs(diff_BA)
    # threshold_ratio = 0.14
    threshold_ratio = 0.3
    # rgb
    real = real_B[:, :, 0:3]
    diff = util.rgb2gray(diff_BA[:, :, 0:3])
    diff = np.uint8(tobinary(real, fake_A[:, :, 0:3], diff, threshold_ratio))
    # kernel = np.ones((3, 3), np.uint8)
    # diff = cv2.dilate(diff, kernel)
    # diff = cv2.erode(diff, kernel)

    arr = diff > 0
    cleaned = morphology.remove_small_objects(arr, min_size=100)
    cleaned = morphology.remove_small_holes(cleaned)
    diff = diff * cleaned

    # remove outer circle
    mask_circle = np.zeros_like(diff)
    w, h, ch = mask_circle.shape
    r = int(h / 2)
    cv2.circle(mask_circle, (r, r), r, (1, 1, 1), thickness=-1)
    diff = diff * mask_circle

    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'blue')
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)

    # read gt
    Atrophy = Image.open(os.path.join('../datasets/PM/PALM-Training400/PALM-Training400-Annotation-Lession/Lesion_Masks/Atrophy', name + '.bmp')).convert('RGB')
    Atrophy = Atrophy.resize((w,h))
    Atrophy = np.array(Atrophy)
    gt = np.zeros_like(Atrophy)
    gt[Atrophy == 0] = 255
    gt[Atrophy == 255] = 0
    # if os.path.exists(os.path.join('../datasets/PM/PALM-Training400/PALM-Training400-Annotation-Lession/Lesion_Masks/Detachment', name + '.bmp')):
    #     Detachment = Image.open(os.path.join('../datasets/PM/PALM-Training400/PALM-Training400-Annotation-Lession/Lesion_Masks/Detachment', name + '.bmp')).convert('RGB')
    #     Detachment = Detachment.resize((w, h))
    #     Detachment = np.array(Detachment)
    #     gt[Detachment == 0] = 255

    img_set = np.concatenate((real, fake_A[:, :, 0:3], diff, stack, gt), axis=1)
    # save
    img_set = np.uint8(img_set)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(img_set, save_path)

def save_AMD_imgs(visuals, image_path, model_name, epoch):
    image_dir = os.path.join('./results', model_name, 'test_' + epoch, 'fakeA')
    # B
    # short_path = ntpath.basename(image_path[1])
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    util.mkdirs(image_dir)
    # real_B = util.tensor2im(visuals['real_B'])
    # fake_A = util.tensor2im(visuals['fake_A'])
    real_B = util.tensor2im(visuals['real_A'])
    fake_A = util.tensor2im(visuals['fake_B'])
    diff_BA = np.int8(real_B) - np.int8(fake_A)
    diff_BA = np.abs(diff_BA)
    threshold_ratio = 13
    # rgb
    real = real_B[:, :, 0:3]
    diff = util.rgb2gray(diff_BA[:, :, 0:3])
    diff = np.uint8(tobinary(real, fake_A[:, :, 0:3], diff, threshold_ratio))
    # kernel = np.ones((3, 3), np.uint8)
    # diff = cv2.dilate(diff, kernel)
    # diff = cv2.erode(diff, kernel)

    arr = diff > 0
    cleaned = morphology.remove_small_objects(arr, min_size=100)
    cleaned = morphology.remove_small_holes(cleaned)
    diff = diff * cleaned

    # remove outer circle
    mask_circle = np.zeros_like(diff)
    w, h, ch = mask_circle.shape
    r = int(h / 2)
    cv2.circle(mask_circle, (r, r), r - 15, (1, 1, 1), thickness=-1)
    diff = diff * mask_circle

    heatmap = cam2img(diff)
    diff = np.uint8(diff)
    diff_on_real = add_diff2img(diff, real,'blue')
    stack = cv2.addWeighted(real, 1, heatmap, 0.5, 0)

    # read gt
    gt = np.zeros_like(diff)
    root = '../datasets/IDRID/Detection and segmentation of lesions from fundus images Training400-Lesion/Training400-Lesion/Lesion_Masks'
    dict = ['drusen','exudate','hemorrhage','others','scar']
    for lesion_name in dict:
        if os.path.exists(os.path.join(root, lesion_name, name + '.bmp')):
            mask = Image.open(os.path.join(root, lesion_name, name + '.bmp')).convert('RGB')
            mask = mask.resize((w,h))
            mask = np.array(mask)
            gt[mask == 0] = 255

    img_set = np.concatenate((real, fake_A[:, :, 0:3], diff, stack, gt), axis=1)
    # save
    img_set = np.uint8(img_set)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(img_set, save_path)

def save_for_oct2octa(visuals, image_path, model_name):
    image_dir = os.path.join('./results', model_name)
    # B
    short_path = ntpath.basename(image_path[1])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathB = os.path.join(image_dir, 'fake_A_visual_res')
    util.mkdirs(save_pathB)
    save_path = os.path.join(save_pathB, image_name)
    fakeB = util.tensor2im_oct2octa(visuals['fake_A'])
    # fakeB = util.tensor2im_oct2octa(visuals['fake_B'])
    util.save_image(fakeB, save_path)

def save_for_oct2octa_from_cyclegan(visuals, image_path, model_name):
    image_dir = os.path.join('./results', model_name)
    # B
    split_path = image_path[1].split('/')
    if len(split_path) == 6:
        case_path = os.path.join(split_path[4], split_path[5][:-4] + '.png')
        save_pathB = os.path.join(image_dir, 'fake_A_from_cyclegan_30000_CNV_Normal')
        util.mkdirs(os.path.join(save_pathB, split_path[4]))
    else:
        case_path = os.path.join(split_path[4], split_path[5], split_path[6][:-4] + '.png')
        save_pathB = os.path.join(image_dir, 'fake_A_from_cyclegan_30000_CNV_Normal')
        util.mkdirs(os.path.join(save_pathB, split_path[4], split_path[5]))
    save_path = os.path.join(save_pathB, case_path)
    fakeB = util.tensor2im(visuals['fake_A'])
    # fakeB = util.tensor2im_oct2octa(visuals['fake_B'])
    util.save_image(fakeB, save_path)

def save_for_oct2octa_for_layerseg(visuals, image_path, model_name):
    image_dir = os.path.join('./results', model_name)
    # B
    split_path = image_path[1].split('/')
    case_path = os.path.join(split_path[4], split_path[5][:-4] + '.png')
    save_pathB = os.path.join(image_dir, 'fake_A_visual_res')
    util.mkdirs(os.path.join(save_pathB, split_path[4]))
    save_path = os.path.join(save_pathB, case_path)
    fakeB = util.tensor2im(visuals['fake_B'])
    # fakeB = util.tensor2im_oct2octa(visuals['fake_B'])
    util.save_image(fakeB, save_path)

def save_for_cnv2oct(visuals, image_path, model_name):
    image_dir = os.path.join('./results', model_name)
    # B
    split_path = image_path[1].split('/')
    if len(split_path) == 6:
        case_path = os.path.join(split_path[4], split_path[5][:-4] + '.png')
        save_pathB = os.path.join(image_dir, 'fake_A_30000')
        util.mkdirs(os.path.join(save_pathB, split_path[4]))
    else:
        case_path = os.path.join(split_path[4], split_path[5], split_path[6][:-4] + '.png')
        save_pathB = os.path.join(image_dir, 'fake_A_30000')
        util.mkdirs(os.path.join(save_pathB, split_path[4], split_path[5]))
    save_path = os.path.join(save_pathB, case_path)
    fakeB = util.tensor2im(visuals['fake_A'])
    # fakeB = util.tensor2im_oct2octa(visuals['fake_B'])
    util.save_image(fakeB, save_path)

def save_for_oct2octa_fakeB(visuals, image_path, model_name):
    image_dir = os.path.join('./results', model_name)
    # B
    short_path = ntpath.basename(image_path[1])
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    save_pathB = os.path.join(image_dir, 'fake_B')
    util.mkdirs(save_pathB)
    save_path = os.path.join(save_pathB, image_name)
    # fakeB = util.tensor2im_oct2octa(visuals['fake_A'])
    fakeB = util.tensor2im_oct2octa(visuals['fake_B'])
    util.save_image(fakeB, save_path)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.display_env = opt.display_env
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, dataset=''):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    if (dataset == 'oct2octa') & ((label == 'fake_A') | (label == 'fake_A_test')):
                    # if dataset == 'oct2octa':
                        image_numpy = util.tensor2im_oct2octa(image)
                    else:
                        image_numpy = util.tensor2im(image)
                    # image_numpy = util.rec2sqr(image_numpy)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_current_accs(self, epoch, counter_ratio, accs):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            accs (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_acc_data'):
            self.plot_acc_data = {'X': [], 'Y': [], 'legend': list(accs.keys())}
        self.plot_acc_data['X'].append(epoch + counter_ratio)
        self.plot_acc_data['Y'].append([accs[k] for k in self.plot_acc_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_acc_data['X'])] * len(self.plot_acc_data['legend']), 1),
                Y=np.array(self.plot_acc_data['Y']),
                opts={
                    'title': self.name + ' acc over time',
                    'legend': self.plot_acc_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'acc'},
                win=self.display_id+3)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def save_vis_plot(self):
        self.vis.save([self.display_env])

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
