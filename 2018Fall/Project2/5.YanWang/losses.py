# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as f
import numpy as np


class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, label):

        intersection = output * label



        den1 = torch.sum(label)



        den2 = torch.sum(output)


        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss
