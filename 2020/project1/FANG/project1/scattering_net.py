# -*- coding: utf-8 -*-

from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D


class ResBlock(nn.Module, ABC):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        """
        :param in_planes: number of input channels
        :param out_planes: number of output channels
        :param stride:
        :param downsample:
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Scattering2dResNet(nn.Module):
    def __init__(self, in_channels, k=2, n=4, num_classes=10):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
            nn.Conv2d(in_channels, self.ichannels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )

        self.layer2 = self._make_layer(ResBlock, 32 * k, n)
        self.layer3 = self._make_layer(ResBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

        self.act_vec = None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.act_vec = self.feature_extraction(x)
        x = self.act_vec
        x = self.fc(x)
        return x

    def feature_extraction(self, x):
        x = x.view(x.size(0), self.K, 7, 7)
        x = self.init_conv(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        # x is the last-layer activations
        return x.view(x.size(0), -1)


recorder = {'training_loss': [],
            'test_loss': [],
            'training_accuracy': [],
            'test_accuracy': [],
            'nc11': [],
            'nc12': [],
            'angle1': [],
            'angle2': [],
            }


def train(model, device, train_loader, optimizer, epoch, scattering):
    model.train()
    ave_loss = 0
    correct = 0
    feature_data = []
    target_data = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        feature_data.append(model.act_vec.cpu().data.numpy())
        target_data.append(target.cpu().numpy())

        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        ave_loss = (batch_idx * ave_loss + loss.item()) / (batch_idx + 1)

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    print('Epoch: {} Train Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

    recorder['training_loss'].append(ave_loss)
    recorder['training_accuracy'].append(correct / len(train_loader.dataset))


    nc = NC(np.concatenate(feature_data), np.concatenate(target_data))
    nc11, nc12 = nc.NC1()
    ang1, ang2 = nc.class_angle()
    recorder['nc11'].append(nc11)
    recorder['nc12'].append(nc12)
    recorder['angle1'].append(ang1)
    recorder['angle2'].append(ang2)


def test(model, device, test_loader, scattering):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    recorder['test_loss'].append(test_loss)
    recorder['test_accuracy'].append(correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class NC:
    def __init__(self, feature_data, target_data):
        self.feature_data = feature_data
        self.target_data = target_data
        self.global_mean = feature_data.mean(axis=0)
        self.class_mean = np.array([feature_data[target_data == i].mean(axis=0) for i in range(10)])

    def NC1(self):
        in_class_feature_dist = np.zeros_like(self.feature_data)
        for i in range(len(self.feature_data)):
            in_class_feature_dist[i] = self.feature_data[i] - self.class_mean[self.target_data[i]]

        in_class_cov = np.matmul(in_class_feature_dist.T, in_class_feature_dist) / len(in_class_feature_dist)

        trace = np.trace(np.matmul(in_class_cov, in_class_cov.T)) / 10

        std_c = np.std(np.linalg.norm(self.class_mean - self.global_mean, axis=1))
        avg_c = np.mean(np.linalg.norm(self.class_mean - self.global_mean, axis=1))
        return trace, std_c / avg_c

    def class_angle(self):
        cos = []
        for i in range(9):
            for j in range(i + 1, 10):
                inner = np.dot(self.class_mean[i] - self.global_mean, self.class_mean[j] - self.global_mean)
                cos.append(
                    inner / np.linalg.norm(self.class_mean[i] - self.global_mean) /
                    np.linalg.norm(self.class_mean[j] - self.global_mean))

        return np.std(cos), np.mean(np.abs(np.array(cos)+1/(10-1)))


if __name__ == '__main__':

    """Train a simple Hybrid Resnet Scattering + CNN model
        scattering 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.
    """
    data_path = '/Users/fanglinjiajie/locals/datasets/'

    N_epoch = 100

    paras = {'mode': 1,
             'width': 2}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if paras['mode'] == 1:
        scattering = Scattering2D(J=2, shape=(28, 28), max_order=1)
        K = 17 * 1
    else:
        scattering = Scattering2D(J=2, shape=(28, 28))
        K = 81 * 1
    if use_cuda:
        scattering = scattering.cuda()

    model = Scattering2dResNet(K, paras['width']).to(device)

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    TRANS = transforms.Compose([transforms.ToTensor()])
    BATCH_SIZE = 100

    train_set = datasets.MNIST(data_path, train=True, transform=TRANS, target_transform=None,
                               download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    test_set = datasets.MNIST(data_path, train=False, transform=TRANS, target_transform=None,
                              download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer
    tr = False
    if tr:
        lr = 0.01
        for epoch in range(0, N_epoch):
            if epoch % 20 == 0:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                            weight_decay=0.0005)
                lr *= 0.5

            train(model, device, train_loader, optimizer, epoch + 1, scattering)
            test(model, device, test_loader, scattering)

        DATA_PATH = '/content/drive/My Drive/deeplearning/'
        torch.save(model.state_dict(), DATA_PATH + 'ScatterNet_MNIST')

    model.load_state_dict(torch.load('ScatterNet_MNIST', map_location=torch.device(device)))

