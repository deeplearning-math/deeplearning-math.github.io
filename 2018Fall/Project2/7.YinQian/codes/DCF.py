import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math
from fb import *


class Conv_DCF(nn.Module):
    r"""Pytorch implementation for 2D DCF Convolution operation.
    Link to ICML paper:
    https://arxiv.org/pdf/1802.04145.pdf


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, optional): Zero-padding added to both sides of
            the input. Default: 0
        num_bases (int, optional): Number of bases for decomposition.
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
                        * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
                        * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 num_bases=6, bias=True, base_grad=False, initializer='FB'):
        super(Conv_DCF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.edge = (kernel_size - 1) / 2
        self.stride = stride
        self.padding = padding
        self.kernel_list = {}
        self.num_bases = num_bases

        assert initializer in ['FB', 'random'], 'Initializer should be either FB or random, other methods are not implemented yet'

        if initializer == 'FB':
            if kernel_size % 2 == 0:
                raise Exception('Kernel size for FB initialization only supports odd number for now.')
            base_np, _, _ = calculate_FB_bases(int((kernel_size - 1) / 2))
            if num_bases > base_np.shape[1]:
                raise Exception('The maximum number of bases for kernel size = %d is %d' % (kernel_size, base_np.shape[1]))
            base_np = base_np[:, :num_bases]
            base_np = base_np.reshape(kernel_size, kernel_size, num_bases)
            base_np = np.expand_dims(base_np.transpose(2, 0, 1), 1)

        else:
            base_np = np.random.random((num_bases, 1, kernel_size, kernel_size)) - 0.5

        self.bases = Parameter(torch.Tensor(base_np), requires_grad=base_grad)

        self.weight = Parameter(torch.Tensor(out_channels, in_channels * num_bases, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        FE_SIZE = inputs.size()
        inputs = inputs.view(FE_SIZE[0] * FE_SIZE[1], 1, FE_SIZE[2], FE_SIZE[3])

        feature = F.conv2d(inputs, self.bases, None, self.stride, self.padding, dilation=1)
        feature = feature.view(FE_SIZE[0], FE_SIZE[1] * self.num_bases,
                               int((FE_SIZE[2] - 2 * self.edge + 2 * self.padding) / self.stride),
                               int((FE_SIZE[3] - 2 * self.edge + 2 * self.padding) / self.stride))

        feature_out = F.conv2d(feature, self.weight, self.bias, 1, 0)

        return feature_out
