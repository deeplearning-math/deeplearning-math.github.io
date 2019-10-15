"""
Regularized inverse of a scattering transform on MNIST
======================================================

Description:
This example trains a convolutional network to invert the scattering transform at scale 2 of MNIST digits.
After only two epochs, it produces a network that transforms a linear interpolation in the scattering space into a
nonlinear interpolation in the image space.

Remarks:
The model after two epochs and the path (which consists of a sequence of images) are stored in the cache directory.
The two epochs take roughly 5 minutes in a Quadro M6000.

Reference:
https://arxiv.org/abs/1805.06621
"""
import sys
sys.path.append('./kymatio-master')

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kymatio import Scattering2D as Scattering
from kymatio.caching import get_cache_dir
from kymatio.datasets import get_dataset_dir



def main():

    transforms_to_apply = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Pixel values should be in [-1,1]
    ])
    
    mnist_dir = get_dataset_dir("MNIST", create=True)
    dataset = datasets.MNIST(mnist_dir, train=False, download=False, transform=transforms_to_apply)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, pin_memory=True)

    fixed_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    fixed_batch = next(iter(fixed_dataloader))
    fixed_batch = fixed_batch[0].float().cuda()

    scattering = Scattering(J=2, shape=(28, 28))
    scattering.cuda()

    
    for _, current_batch in enumerate(dataloader):
        batch_images = Variable(current_batch[0]).float().cuda()
        batch_scattering = scattering(batch_images).squeeze(1)
        
        print(batch_scattering.shape)
        exit()


if __name__ == '__main__':
    main()
