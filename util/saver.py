import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
# from scipy.misc import imresize

def save_images(image,image_path,save_dir,suffix):

    short_path = ntpath.basename(image_path)
    name = os.path.splitext(short_path)
    image_name = '%s_%s.png' % (name[0], suffix)
    save_path = os.path.join(save_dir, image_name)

    if not os.path.exists(save_dir):
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(save_dir)

    im = util.tensor2im(image)
    util.save_image(im, save_path)

class Saver():
    def __init__(self, opt):
        pass