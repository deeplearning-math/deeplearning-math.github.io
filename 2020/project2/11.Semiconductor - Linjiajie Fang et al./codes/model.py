import numpy as np
from abc import ABC
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class StratifiedConv(nn.Module, ABC):
    def __init__(self, in_planes, out_planes):
        super(StratifiedConv, self).__init__()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.conv_1 = \
            nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=False)
        self.conv_3 = \
            nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.conv_9 = \
            nn.Conv2d(in_planes, out_planes, kernel_size=(9, 9), stride=1, padding=(4, 4), bias=False)
        self.conv_27 = \
            nn.Conv2d(in_planes, out_planes, kernel_size=(27, 27), stride=1, padding=(13, 13), bias=False)
        self.conv_81 = \
            nn.Conv2d(in_planes, out_planes, kernel_size=(81, 81), stride=1, padding=(40, 40), bias=False)
        self.conv_merge = nn.Sequential(
            nn.BatchNorm2d(out_planes * 4),
            nn.Conv2d(4 * out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(out_planes),
            # nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv_3(x)
        x2 = self.conv_9(x)
        x3 = self.conv_27(x)
        x4 = self.conv_81(x)
        x = self.conv_merge(torch.cat([x1, x2, x3, x4], dim=1))
        return x


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


class ResNet(nn.Module):
    """
    k gives the multiplication factor of the hidden layers. larger k -> larger nets
    n gives the depth: number of residue blocks in each layer
    """

    def __init__(self, in_channels, out_dim, k=2, n=8):
        super(ResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.init_conv = nn.Sequential(
            StratifiedConv(in_channels,self.ichannels),
            # nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
            # nn.Conv2d(in_channels, self.ichannels,
            #           kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU()
        )
        self.avgpool1 = nn.AdaptiveAvgPool2d(64)
        self.layer2 = self._make_layer(ResBlock, 32 * k, n)
        self.avgpool2 = nn.AdaptiveAvgPool2d(8)
        self.layer3 = self._make_layer(ResBlock, 64 * k, n)
        self.avgpool3 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64 * k, 32)
        self.fc2 = nn.Linear(32, out_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        :param block: The residual block
        :param planes: the output channel
        :param blocks: number of inner blocks
        :param stride:
        :return:
        """
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

    def num_paras(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.avgpool1(self.init_conv(x))
        x = self.avgpool2(self.layer2(x))
        x = self.avgpool3(self.layer3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return torch.sigmoid(x)

    def activation_vector(self, x):
        """
        :return: 32-dim vector
        """
        x = self.avgpool1(self.init_conv(x))
        x = self.avgpool2(self.layer2(x))
        x = self.avgpool3(self.layer3(x))
        return F.relu(self.fc1(x.view(x.size(0), -1)))


class YOLO(nn.Module):
    def __init__(self, in_channels, grids=8, B=1, k=2, n=4):
        super(YOLO, self).__init__()
        self.grids = grids
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
            nn.Conv2d(in_channels, self.ichannels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU()
        )
        self.maxpool1 = nn.AdaptiveMaxPool2d(64)
        self.layer2 = self._make_layer(ResBlock, 32 * k, n)
        self.maxpool2 = nn.AdaptiveMaxPool2d(16)
        self.layer3 = self._make_layer(ResBlock, 64 * k, n)
        self.maxpool3 = nn.AdaptiveMaxPool2d(grids)
        self.fc1 = nn.Linear(64 * k * grids * grids, 1024)
        self.fc2 = nn.Linear(1024, grids * grids * B * 6)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        :param block: The residual block
        :param planes: the output channel
        :param blocks: number of inner blocks
        :param stride:
        :return:
        """
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

    def num_paras(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.maxpool1(self.init_conv(x))
        x = self.maxpool2(self.layer2(x))
        x = self.maxpool3(self.layer3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x).view(-1, self.grids, self.grids, 6)
        return x.sigmoid()


class yolo_loss(nn.Module):
    def __init__(self, S=8, B=1):
        """

        :param S: gird num
        :param B: num of predicted boxes
        """
        super(yolo_loss, self).__init__()
        self.S = S
        self.B = B

    def forward(self, pred_tensor, target_tensor):
        """
        x1, y1 be the top left point
        c: IOU
        p: 1 or 0
        :param pred_tensor: (-1, 8, 8, 6) [x1,y1,w,h,c,p] w = dx, h = dy
        :param target_tensor: (-1, 8, 8, 6) [x1,y1,w,h,c,p] normalized to [0,1] (-1 * 4)
        :return:
        """
        # coordiate loss
        obj_mask = target_tensor[:, :, :, 4] > 0
        noobj_mask = target_tensor[:, :, :, 4] == 0
        loss = F.mse_loss(pred_tensor[obj_mask][:, :2], target_tensor[obj_mask][:, :2])
        loss += F.mse_loss(torch.sqrt(pred_tensor[obj_mask][:, 2:4]), torch.sqrt(target_tensor[obj_mask][:, 2:4]))
        # confidence loss 1*IOU
        loss = loss * 5 + F.mse_loss(pred_tensor[obj_mask].view(-1, 6)[:, 4], target_tensor[obj_mask].view(-1, 6)[:, 4])
        # existence loss: 0 or 1
        loss += F.mse_loss(pred_tensor[obj_mask].view(-1, 6)[:, 5], target_tensor[obj_mask].view(-1, 6)[:, 5])
        # no object loss
        loss += 0.5 * F.mse_loss(pred_tensor[noobj_mask].view(-1, 6)[:, 4], target_tensor[noobj_mask].view(-1, 6)[:, 4])

        return loss

        # center_x, center_y = (target_tensor[:, 0] + target_tensor[:, 2]) / 2, \
        #                      (target_tensor[:, 1] + target_tensor[:, 3]) / 2
        # center_x, center_y = center_x.sigmoid(), center_y.sigmoid()
        # x_row, y_row = (8 * center_x).long(), (8 * center_y).long()
        #
        # coord = pred_tensor[:, :, :, :, :4]
        # target[:, :, :, :, :4] = target_tensor

    @staticmethod
    def get_points(box):
        return [(box[0], box[1]), (box[2], box[3]), (box[0], box[3]), (box[2], box[1])]

    @staticmethod
    def get_box(points):
        # four points
        x, y = set(), set()
        for p in points:
            x.add(p[0])
            y.add(p[1])
        return min(x), min(y), max(x), max(y)

    @staticmethod
    def get_area(box):
        return abs((box[0] - box[2]) * (box[1] - box[3]))
