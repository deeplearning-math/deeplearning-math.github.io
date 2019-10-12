from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.models as models
from torchvision.datasets import MNIST
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
import os
import numpy as np

original_model = models.resnet18(pretrained=True)

class resnet18Conv5(nn.Module):
    def __init__(self):
        super(resnet18Conv5, self).__init__()

        self.features = nn.Sequential(
            # stop at conv4
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            *list(original_model.children())[:-3]
        )
    def forward(self, x):
        x = self.features(x)
        return x

def get_data_loaders(train_batch_size, val_batch_size):
    mnist = MNIST(download=True, train=True, root=".").train_data.float()
    
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=False)

    val_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

def get_feature_map(idx):

    model = resnet18Conv5().cuda()

    mnist = MNIST(download=True, train=True, root=".").train_data.float()
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])    
    mnist_dataset = MNIST(download=True, root=".", transform=data_transform, train=True)
    
    print(len(mnist))
    data = mnist_dataset[idx]
    print("data shape = ", data[0].shape)# data shape 1x224x224\
    print("data label = ", data[1])# label number
    X = data[0].cuda()
    X = X.unsqueeze(0)
    outputs = model(X)
    outputs = outputs.detach().cpu().numpy()
    print(outputs.shape)
    return outputs


if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
    os.system('rm tmp')

    feat_map = get_feature_map(idx=0) # shape: [1, 256, 14, 14]
    #print(feat_map)