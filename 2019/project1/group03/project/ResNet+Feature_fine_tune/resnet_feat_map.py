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

class resnet18fc1000(nn.Module):
    def __init__(self):
        super(resnet18fc1000, self).__init__()

        self.features = nn.Sequential(
            # stop at conv4
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            *list(original_model.children())[:-1]
        )
        # self.fc = nn.Sequential(*list(original_model.children())[-1])
        self.fc = list(original_model.children())[-1]
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_feature_map(idx):

    model = resnet18fc1000().cuda()

    mnist = MNIST(download=True, train=True, root=".").train_data.float()
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])    
    mnist_dataset = MNIST(download=True, root=".", transform=data_transform, train=True)
    data = mnist_dataset[idx]

    X = data[0].cuda()
    X = X.unsqueeze(0)
    
    outputs = model(X)

    return outputs

if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
    os.system('rm tmp')
    
    feat_map = get_feature_map(idx=0) # shape: [1, 256, 14, 14]


