import os.path
from data.base_dataset import BaseDataset, get_transform, DR_flip
from data.image_folder import make_dataset
from PIL import Image
import random
import csv
import numpy as np
import ntpath
import torch
import sys


class AMDDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.test_for_traindata = self.opt.test_for_traindata

        # augment:
        # 0:none   1:flip  2:clock5   3:anticlock5
        # 4:offset-up5   5:offset-down5   6:offset-left5   7:offset-right5
        if opt.isTrain:
            self.augment = opt.augment
        else:
            self.augment = False

        # A
        self.A_paths = make_dataset(os.path.join(opt.dataroot, 'Classification of AMD and non-AMD fundus images iChallenge-AMD-Training400/Training400/Non-AMD'), opt.max_dataset_size)
        self.A_paths = self._sort(self.A_paths)
        len_A = len(self.A_paths)
        split_loc = int(len_A / 5)
        self.test_A_paths = self.A_paths[:split_loc]
        self.train_A_paths = self.A_paths[split_loc:]
        # B
        self.B_paths = make_dataset(os.path.join(opt.dataroot, 'Classification of AMD and non-AMD fundus images iChallenge-AMD-Training400/Training400/AMD'), opt.max_dataset_size)
        self.B_paths = self._sort(self.B_paths)
        len_B = len(self.B_paths)
        split_loc = int(len_B / 5)
        self.test_B_paths = self.B_paths[:split_loc]
        self.train_B_paths = self.B_paths[split_loc:]

        self.train_A_size = len(self.train_A_paths)  # get the size of dataset A
        self.train_B_size = len(self.train_B_paths)  # get the size of dataset B
        self.test_A_size = len(self.test_A_paths)  # get the size of dataset A
        self.test_B_size = len(self.test_B_paths)  # get the size of dataset B

        if self.augment:
            self.train_A_paths = [val for val in self.train_A_paths for i in range(8)]
            self.train_B_paths = [val for val in self.train_B_paths for i in range(8)]
            aug_labels = list(range(8))
            self.train_A_aug_labels = aug_labels * self.train_A_size
            self.train_B_aug_labels = aug_labels * self.train_B_size
            self.train_A_size = len(self.train_A_paths)  # get the size of dataset A
            self.train_B_size = len(self.train_B_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def _sort(self, list):
        '''
        list :待排列数组
        b:数字前一个字符
        a;数字后一个字符
        '''
        list.sort(key=lambda x: int(x.split('/')[-1].split('.')[-2][1:]))
        return list

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.isTrain | self.test_for_traindata:
            if self.opt.serial_batches:  # make sure index is within then range
                train_index_A = index % self.train_A_size
                train_index_B = index % self.train_B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                train_index_A = random.randint(0, self.train_A_size - 1)
                train_index_B = random.randint(0, self.train_B_size - 1)
            train_A_path = self.train_A_paths[train_index_A]
            train_B_path = self.train_B_paths[train_index_B]
            train_A_img = Image.open(train_A_path).convert('RGB')
            train_B_img = Image.open(train_B_path).convert('RGB')
            # apply image transformation
            if self.augment:
                self.transform_A = get_transform(self.opt, grayscale=False,
                                                 oct2octa_trans=self.train_A_aug_labels[train_index_A])
                self.transform_B = get_transform(self.opt, grayscale=False,
                                                 oct2octa_trans=self.train_B_aug_labels[train_index_B])
            train_A = self.transform_A(train_A_img)
            train_B = self.transform_B(train_B_img)

        if self.opt.serial_batches:  # make sure index is within then range
            test_index_A = index % self.test_A_size
            test_index_B = index % self.test_B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            test_index_A = random.randint(0, self.test_A_size - 1)
            test_index_B = random.randint(0, self.test_B_size - 1)
        test_A_path = self.test_A_paths[test_index_A]
        test_B_path = self.test_B_paths[test_index_B]
        test_A_img = Image.open(test_A_path).convert('RGB')
        test_B_img = Image.open(test_B_path).convert('RGB')
        # apply image transformation
        test_A = self.transform_A(test_A_img)
        test_B = self.transform_B(test_B_img)

        if self.opt.isTrain | self.test_for_traindata:
            return {'train_A': train_A, 'train_B': train_B, 'train_A_paths': train_A_path,
                    'train_B_paths': train_B_path, 'test_A': test_A, 'test_B': test_B, 'test_A_paths': test_A_path,
                    'test_B_paths': test_B_path}
        else:
            return {'test_A': test_A, 'test_B': test_B, 'test_A_paths': test_A_path,
                    'test_B_paths': test_B_path}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.opt.isTrain | self.test_for_traindata:
            return max(self.train_A_size, self.train_B_size)
        else:
            return max(self.test_A_size, self.test_B_size)