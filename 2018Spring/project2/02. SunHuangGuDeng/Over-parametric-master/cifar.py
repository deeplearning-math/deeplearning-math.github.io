# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import os
import sys
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10(Dataset):
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, image_mode='normal', label_mode='normal'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            if image_mode == 'normal':
                self.train_data = self.train_data
            elif image_mode == 'random':
                self.train_data = np.random.permutation(
                    self.train_data.reshape((50000, -1)).transpose()).transpose().reshape((50000, 32, 32, 3))
            elif image_mode == 'shuffled':
                np.random.seed(0)
                index = np.random.permutation(range(32 * 32 * 3))
                self.train_data = self.train_data.reshape((50000, -1))[:, index].reshape((50000, 32, 32, 3))
            elif image_mode == 'gaussian':
                mean = np.mean(np.mean(self.train_data.reshape((50000, -1)), axis=1))
                std = np.mean(np.std(self.train_data.reshape((50000, -1)), axis=1))
                self.train_data = np.random.rand(50000, 32, 32, 3) * std + mean

            if label_mode == 'normal':
                self.train_labels = self.train_labels
            elif label_mode == 'random':
                self.train_labels = np.random.choice(range(10), 50000)
            elif label_mode.startswith('partially'):
                p = float(label_mode.split('-')[1])
                self.train_labels = [y if np.random.uniform() < p else np.random.randint(0, 10) for y in
                                     self.train_labels]


        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
