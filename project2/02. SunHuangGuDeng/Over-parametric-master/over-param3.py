# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import os
import sys
import pickle
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from cifar import CIFAR10
from models import basic_cnn
from utils import AverageMeter, accuracy, Logger

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

root = './'
mode_set = [('normal', 'normal'), ('normal', 'random'), ('normal', 'partially-0.1'),
            ('normal', 'partially-0.3'), ('normal', 'partially-0.5'), ('normal', 'partially-0.7'),
            ('normal', 'partially-0.9'), ('random', 'normal'), ('shuffled', 'normal')]

error = []

for (mode1, mode2) in mode_set:
    training_dataset = CIFAR10(root, train=True, image_mode=mode1, label_mode=mode2)
    testing_dataset = CIFAR10(root, train=False)

    training_data, training_labels = training_dataset.train_data.reshape(
        (50000, -1)) / 255, training_dataset.train_labels
    testing_data, testing_labels = testing_dataset.test_data.reshape((10000, -1)) / 255, testing_dataset.test_labels

    n = 1
    pipline = Pipeline([('knn', KNeighborsClassifier(n_neighbors=n, n_jobs=-1))])
    pipline.fit(training_data, training_labels)
    testing_predictions = pipline.predict(testing_data)

    testing_error = np.mean(testing_predictions != testing_labels)

    print('{}-{} : testing error {}'.format(mode1, mode2, testing_error))

    error.append(testing_error)
