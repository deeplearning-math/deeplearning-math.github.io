import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
from padding_same import Conv2d_same_padding
import mrcfile
import torch.nn as nn
import torch.nn.functional as F
from utils import conv2d, batch_norm, Identity_block_for_D, Identity_block_for_G

import utils
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import mrcfile
import os
import h5py
import cv2
import sys
import numpy as np
from skimage.exposure import rescale_intensity
import logging
import scipy.io as sio
import matplotlib.image as mpimg
import random


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def make_layer_G(filter):
    layer = nn.Sequential(
        nn.Conv2d(filter, filter, 1, 1, 0),
        nn.BatchNorm2d(filter),
        nn.ReLU(),
        Conv2d_same_padding(filter, filter, kernel_size=4, stride=1),
        nn.BatchNorm2d(filter),
        nn.ReLU(),
        nn.Conv2d(filter, filter, 1, 1, 0),
        nn.BatchNorm2d(filter),
    )
    return layer


def make_layer_D(filter):
    layer = nn.Sequential(
        nn.Conv2d(filter, filter, 1, 1, 0),
        nn.BatchNorm2d(filter),
        nn.ReLU(),
        Conv2d_same_padding(filter, filter, kernel_size=4, stride=1),
        nn.BatchNorm2d(filter),
        nn.ReLU(),
        nn.Conv2d(filter, filter, 1, 1, 0),
        nn.BatchNorm2d(filter),
    )
    return layer


class generator(nn.Module):
    def __init__(self, d):
        super(generator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.layere1a = make_layer_G(64)
        self.layere1b = make_layer_G(64)
        self.layere1c = make_layer_G(64)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.layere2a = make_layer_G(128)
        self.layere2b = make_layer_G(128)
        self.layere2c = make_layer_G(128)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.layere3a = make_layer_G(256)
        self.layere3b = make_layer_G(256)
        self.layere3c = make_layer_G(256)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.layere4a = make_layer_G(512)
        self.layere4b = make_layer_G(512)
        self.layere4c = make_layer_G(512)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.layere5a = make_layer_G(512)
        self.layere5b = make_layer_G(512)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.layerd1a = make_layer_G(512)
        self.layerd1b = make_layer_G(512)
        self.conv8 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.conv8_bn = nn.BatchNorm2d(d * 8)
        self.layerd2a = make_layer_G(512)
        self.layerd2b = make_layer_G(512)
        self.layerd2c = make_layer_G(512)
        self.conv9 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.conv9_bn = nn.BatchNorm2d(d * 4)
        self.layerd3a = make_layer_G(256)
        self.layerd3b = make_layer_G(256)
        self.layerd3c = make_layer_G(256)
        self.conv10 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.conv10_bn = nn.BatchNorm2d(d * 2)
        self.layerd4a = make_layer_G(128)
        self.layerd4b = make_layer_G(128)
        self.layerd4c = make_layer_G(128)
        self.conv11 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.conv11_bn = nn.BatchNorm2d(d)
        self.layerd5a = make_layer_G(64)
        self.layerd5b = make_layer_G(64)
        self.layerd5c = make_layer_G(64)
        self.conv12 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        self.conv12_bn = nn.BatchNorm2d(1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, img):
        # encoder
        e1 = self.conv1_bn(self.conv1(img))
        e10 = F.relu(e1)
        e1a = F.relu(torch.add(self.layere1a(e10), 1, e10))
        e1b = F.relu(torch.add(self.layere1b(e1a), 1, e1a))
        e1c = F.relu(torch.add(self.layere1c(e10), 1, e1b))

        e2 = self.conv2_bn(self.conv2(e1c))
        e20 = F.relu(e2)
        e2a = F.relu(torch.add(self.layere2a(e20), 1, e20))
        e2b = F.relu(torch.add(self.layere2b(e2a), 1, e2a))
        e2c = F.relu(torch.add(self.layere2c(e2b), 1, e2b))

        e3 = self.conv3_bn(self.conv3(e2c))
        e30 = F.relu(e3)
        e3a = F.relu(torch.add(self.layere3a(e30), 1, e30))
        e3b = F.relu(torch.add(self.layere3b(e3a), 1, e3a))
        e3c = F.relu(torch.add(self.layere3c(e3b), 1, e3b))

        e4 = self.conv4_bn(self.conv4(e3c))
        e40 = F.relu(e4)
        e4a = F.relu(torch.add(self.layere4a(e40), 1, e40))
        e4b = F.relu(torch.add(self.layere4b(e4a), 1, e4a))
        e4c = F.relu(torch.add(self.layere4c(e4b), 1, e4b))

        e5 = self.conv5_bn(self.conv5(e4c))
        e50 = F.relu(e5)
        e5a = F.relu(torch.add(self.layere5a(e50), 1, e50))
        e5b = F.relu(torch.add(self.layere5b(e5a), 1, e5a))
        # e5c = Identity_block_for_G(e5b, 512)

        e6 = self.conv6_bn(self.conv6(e5b))
        e60 = F.relu(e6)

        # decoder

        d10 = F.relu(torch.add(self.conv7_bn(self.conv7(e60)), 1, e5))
        d1a = F.relu(torch.add(self.layerd1a(d10), 1, d10))
        d1b = F.relu(torch.add(self.layerd1b(d1a), 1, d1a))
        # d1c = Identity_block_for_G(d1b, 512)

        d20 = F.relu(torch.add(self.conv8_bn(self.conv8(d1b)), 1, e4))
        d2a = F.relu(torch.add(self.layerd2a(d20), 1, d20))
        d2b = F.relu(torch.add(self.layerd2b(d2a), 1, d2a))
        d2c = F.relu(torch.add(self.layerd2c(d2b), 1, d2b))

        d30 = F.relu(torch.add(self.conv9_bn(self.conv9(d2c)), 1, e3))
        d3a = F.relu(torch.add(self.layerd3a(d30), 1, d30))
        d3b = F.relu(torch.add(self.layerd3b(d3a), 1, d3a))
        d3c = F.relu(torch.add(self.layerd3c(d3b), 1, d3b))

        d40 = F.relu(torch.add(self.conv10_bn(self.conv10(d3c)), 1, e2))
        d4a = F.relu(torch.add(self.layerd4a(d40), 1, d40))
        d4b = F.relu(torch.add(self.layerd4b(d4a), 1, d4a))
        d4c = F.relu(torch.add(self.layerd4c(d4b), 1, d4b))

        d50 = F.relu(torch.add(self.conv11_bn(self.conv11(d4c)), 1, e1))
        d5a = F.relu(torch.add(self.layerd5a(d50), 1, d50))
        d5b = F.relu(torch.add(self.layerd5b(d5a), 1, d5a))
        d5c = F.relu(torch.add(self.layerd5c(d5b), 1, d5b))

        d60 = self.conv12_bn(self.conv12(d5c))
        return F.tanh(d60)


class discriminator(nn.Module):
    # initializers
    def __init__(self, d):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2, d, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.layerf1a = make_layer_D(64)
        self.layerf1b = make_layer_D(64)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.layerf2a = make_layer_D(128)
        self.layerf2b = make_layer_D(128)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.layerf3a = make_layer_D(256)
        self.layerf3b = make_layer_D(256)
        self.layerf3c = make_layer_D(256)
        self.conv4 = Conv2d_same_padding(256, 512, kernel_size=4, stride=1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.layerf4a = make_layer_D(512)
        self.layerf4b = make_layer_D(512)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 1, 0)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, img, cond):
        x = torch.cat([img, cond], 1)
        f10 = F.relu(self.conv1_bn(self.conv1(x)))  # 128
        f1a = F.relu(torch.add(self.layerf1a(f10), 1, f10))
        f1b = F.relu(torch.add(self.layerf1b(f1a), 1, f1a))
        f20 = F.relu(self.conv2_bn(self.conv2(f1b)))  # 64
        f2a = F.relu(torch.add(self.layerf2a(f20), 1, f20))
        f2b = F.relu(torch.add(self.layerf2b(f2a), 1, f2a))
        f30 = F.relu(self.conv3_bn(self.conv3(f2b)))  # 32
        f3a = F.relu(torch.add(self.layerf3a(f30), 1, f30))
        f3b = F.relu(torch.add(self.layerf3b(f3a), 1, f3a))
        f3c = F.relu(torch.add(self.layerf3c(f3b), 1, f3b))
        f40 = F.relu(self.conv4_bn(self.conv4(f3c)))  # 32
        f4a = F.relu(torch.add(self.layerf4a(f40), 1, f40))
        f4b = F.relu(torch.add(self.layerf4b(f4a), 1, f4a))
        f50 = F.relu(self.conv5_bn(self.conv5(f4b)))  # 29
        f60 = self.conv6(f50)  # 26

        return f60
