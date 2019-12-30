import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class DriveData(Dataset):
    __xs = None
    __ys = None

    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.__xs = X
        self.__ys = y

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.fromarray(self.__xs[index,:,:])
        img.convert('L')
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.LongTensor(np.asarray(self.__ys[index]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)