import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

  def __init__(self, num_channels):
    super(CNN, self).__init__()
    # 5 convolutional layers without pooling, number of channels per layer defined by user
    self.conv1 = nn.Conv2d(3, num_channels, 5)
    self.conv_bn1 = nn.BatchNorm2d(num_channels)
    self.conv2 = nn.Conv2d(num_channels, num_channels, 5)
    self.conv_bn2 = nn.BatchNorm2d(num_channels)
    self.conv3 = nn.Conv2d(num_channels, num_channels, 5)
    self.conv_bn3 = nn.BatchNorm2d(num_channels)
    self.conv4 = nn.Conv2d(num_channels, num_channels, 5)
    self.conv_bn4 = nn.BatchNorm2d(num_channels)
    self.conv5 = nn.Conv2d(num_channels, num_channels, 5)
    self.conv_bn5 = nn.BatchNorm2d(num_channels)
    # 2 fully connected layers
    self.fc1 = nn.Linear(num_channels * 5 * 5, num_channels * 5)
    self.fc_bn1 = nn.BatchNorm1d(num_channels * 5)
    self.fc2 = nn.Linear(num_channels * 5, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = x.view(-1, self.num_flat_features)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
      num_features *= s
    return num_features
