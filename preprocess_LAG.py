import os
from PIL import Image
import numpy as np
import shutil
import csv
import util.util

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def split_train_test(srcdir, dstdir):
    train_ratio = 0.75
    srcA = os.path.join(srcdir, 'non_glaucoma')
    srcB = os.path.join(srcdir, 'suspicious_glaucoma')

    # non_glaucoma A
    dst_im_train = os.path.join(dstdir, 'image', 'trainA')
    dst_im_test = os.path.join(dstdir, 'image', 'testA')
    util.util.mkdir(dst_im_train)
    util.util.mkdir(dst_im_test)
    dst_am_train = os.path.join(dstdir, 'attention_map', 'trainA')
    dst_am_test = os.path.join(dstdir, 'attention_map', 'testA')
    util.util.mkdir(dst_am_train)
    util.util.mkdir(dst_am_test)
    cnt = 0
    for root, _, fnames in sorted(os.walk(os.path.join(srcA, 'image'))):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if cnt < np.int(np.floor(3143 * train_ratio)):
                    # train
                    shutil.copy(path, dst_im_train)
                    shutil.copy(os.path.join(srcA, 'attention_map', fname), dst_am_train)
                else:
                    # test
                    shutil.copy(path, dst_im_test)
                    shutil.copy(os.path.join(srcA, 'attention_map', fname), dst_am_test)
                cnt += 1

    # suspicious_glaucoma B
    dst_im_train = os.path.join(dstdir, 'image', 'trainB')
    dst_im_test = os.path.join(dstdir, 'image', 'testB')
    util.util.mkdir(dst_im_train)
    util.util.mkdir(dst_im_test)
    dst_am_train = os.path.join(dstdir, 'attention_map', 'trainB')
    dst_am_test = os.path.join(dstdir, 'attention_map', 'testB')
    util.util.mkdir(dst_am_train)
    util.util.mkdir(dst_am_test)
    cnt = 0
    for root, _, fnames in sorted(os.walk(os.path.join(srcB, 'image'))):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if cnt < np.int(np.floor(1711 * train_ratio)):
                    # train
                    shutil.copy(path, dst_im_train)
                    shutil.copy(os.path.join(srcB, 'attention_map', fname), dst_am_train)
                else:
                    # test
                    shutil.copy(path, dst_im_test)
                    shutil.copy(os.path.join(srcB, 'attention_map', fname), dst_am_test)
                cnt += 1

if __name__ == '__main__':

    srcdir = '../datasets/LAG_database_part_1'
    dstdir = '../datasets/LAG_database_part_1/splited_all'

    split_train_test(srcdir, dstdir)

    print('ciao')