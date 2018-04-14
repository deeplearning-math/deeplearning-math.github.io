# -*- coding: utf-8 -*-
__author__ = 'huangyf'

from torch import nn


class CNN(nn.Module):
    def __init__(self, channels):
        super(CNN, self).__init__()
        self.features = self._make_layers(channels)
        self.classifier = nn.Linear(channels, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, channels):
        layers = []
        in_channels = 3
        for i in range(5):
            if i == 0:
                layers += [nn.Conv2d(in_channels, channels, 3, 2, 1),
                           nn.BatchNorm2d(channels),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(channels, channels, 3, 2, 1),
                           nn.BatchNorm2d(channels),
                           nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)
