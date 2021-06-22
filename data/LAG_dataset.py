import os.path
from data.base_dataset import BaseDataset, get_transform, DR_flip
from data.image_folder import make_dataset
from PIL import Image
import random
import csv
import numpy as np
import ntpath


class LAGDataset(BaseDataset):
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

        folds_idx = {
            1: {'A': 0, 'B': 0},
            2: {'A': 628, 'B': 342},
            3: {'A': 1256, 'B': 684},
            4: {'A': 1884, 'B': 1026},
            5: {'A': 2512, 'B': 1368},
            6: {'A': 3143, 'B': 1711},
        }

        self.test_for_traindata = self.opt.test_for_traindata

        # A
        self.train_A_paths = []
        with open(os.path.join(opt.dataroot, 'fold5', 'A_list.txt')) as f:
            for line in f:
                self.train_A_paths.append(os.path.join(opt.dataroot, line.strip('\n')))
        self.test_A_paths = self.train_A_paths[folds_idx[opt.cur_fold]['A'] : folds_idx[opt.cur_fold+1]['A']]
        del self.train_A_paths[folds_idx[opt.cur_fold]['A'] : folds_idx[opt.cur_fold+1]['A']]

        # B
        self.train_B_paths = []
        with open(os.path.join(opt.dataroot, 'fold5', 'B_list.txt')) as f:
            for line in f:
                self.train_B_paths.append(os.path.join(opt.dataroot, line.strip('\n')))
        self.test_B_paths = self.train_B_paths[folds_idx[opt.cur_fold]['B'] : folds_idx[opt.cur_fold+1]['B']]
        del self.train_B_paths[folds_idx[opt.cur_fold]['B'] : folds_idx[opt.cur_fold+1]['B']]

        self.train_A_size = len(self.train_A_paths)  # get the size of dataset A
        self.train_B_size = len(self.train_B_paths)  # get the size of dataset B
        self.test_A_size = len(self.test_A_paths)  # get the size of dataset A
        self.test_B_size = len(self.test_B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))


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
            train_A_path = self.train_A_paths[index % self.train_A_size]  # make sure index is within then range
            if self.opt.serial_batches:  # make sure index is within then range
                train_index_B = index % self.train_B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                train_index_B = random.randint(0, self.train_B_size - 1)
            train_B_path = self.train_B_paths[train_index_B]
            train_A_img = DR_flip(Image.open(train_A_path).convert('RGB'),train_A_path)
            train_B_img = DR_flip(Image.open(train_B_path).convert('RGB'),train_A_path)
            # apply image transformation
            train_A = self.transform_A(train_A_img)
            train_B = self.transform_B(train_B_img)

        test_A_path = self.test_A_paths[index % self.test_A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            test_index_B = index % self.test_B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            test_index_B = random.randint(0, self.test_B_size - 1)
        test_B_path = self.test_B_paths[test_index_B]
        test_A_img = DR_flip(Image.open(test_A_path).convert('RGB'), test_A_path)
        test_B_img = DR_flip(Image.open(test_B_path).convert('RGB'), test_A_path)
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