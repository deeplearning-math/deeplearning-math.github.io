import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_ as xavier_normal_
from torch.nn.init import kaiming_normal_ as kaiming_normal_
import numpy as np


## Basic Resnet Block for upsampling(gener)/ downsampling(disc)/ unchanged(disc)
class ResBlock(nn.Module):

    def __init__(self, input_c, output_c, kernal_size=3, pad=1, resample=None):
        super(ResBlock, self).__init__()
        if resample == 'up':
            self.conv_1 = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(input_c, output_c, kernal_size, 1, pad))
            self.conv_2 = nn.Conv2d(output_c, output_c, kernal_size, 1, pad)
            self.conv_shortcut = nn.Sequential( ## upsample - 1 by 1 conv layer
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(input_c, output_c, 1, 1, 0))

        if resample == 'down':
            self.conv_1 = nn.Conv2d(input_c, input_c, kernal_size, 1, pad)
            self.conv_2 = nn.Sequential(
                nn.Conv2d(input_c, output_c, kernal_size, 1, pad),
                nn.AvgPool2d(2,2))
            self.conv_shortcut = nn.Sequential(
                nn.Conv2d(input_c, output_c, 1, 1, 0), ## shortcut we use 1*1 conv layer
                nn.AvgPool2d(2,2))

        # if resample == None:
        #     self.conv_1 = nn.Conv2d(input_c, output_c, kernal_size, 1, pad)
        #     self.conv_2 = nn.Conv2d(input_c, output_c, kernal_size, 1, pad)
        #     if input_c == output_c:
        #         self.conv_shortcut = lambda x: x
        #     else:
        #         print('Warning: resample=None & input_c!=output_c')
        #         self.conv_shortcut = nn.Conv2d(input_c, output_c, 1, 1, 0)

        self.map1 = nn.Sequential(
            nn.BatchNorm2d(input_c),
            nn.LeakyReLU(.2))

        if resample == 'up':
            self.map1 = nn.Sequential(
                nn.BatchNorm2d(input_c),
                nn.LeakyReLU(.2))
            self.map2 = nn.Sequential(
                nn.BatchNorm2d(output_c),
                nn.LeakyReLU(.2))
        if resample == 'down':
            self.map1 = nn.LeakyReLU(.2)

            self.map2 = nn.LeakyReLU(.2)


    def forward(self, x):

        x_shortcut = self.conv_shortcut(x)
        x = self.map1(x)
        x = self.conv_1(x)
        x = self.map2(x)
        x = self.conv_2(x)

        return x_shortcut + x

class Generator(nn.Module):
    def __init__(self, z_dim=128, d=64):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.d = d
        self.resblock_1 = ResBlock(input_c=8*self.d, output_c=4*self.d, kernal_size=3, resample='up') # 512,4,4 -> 256,8,8
        self.resblock_2 = ResBlock(4*self.d, 2*self.d, 3, resample='up') #256,8,8 -> 128,16,16
        self.resblock_3 = ResBlock(2*self.d, self.d ,3, resample='up') #128,16,16 -> 64,32,32

        self.proj = nn.Linear(self.z_dim, 8*self.d*4*4)
        self.map = nn.Sequential(
            #nn.BatchNorm2d(self.d),
            nn.LeakyReLU(.2),
            nn.Conv2d(self.d, 3, 3, 1, 1))
        self.tanh = nn.Tanh()

    def forward(self, z):

        x = self.proj(z).reshape(-1, 8*self.d, 4, 4)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)

        x = self.map(x)
        img = self.tanh(x)

        return img


class Discriminator_Res(nn.Module):

    def __init__(self, d=64):
        super(Discriminator_Res, self).__init__()

        self.d = d
        self.map = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(.2)) # 3,32,32 -> 64,32,32
        self.resblock_1 = ResBlock(input_c=self.d, output_c=2*self.d, resample='down') #64, 32, 32 -> 128,16,16
        self.resblock_2 = ResBlock(2*self.d, 4*self.d, resample='down') #128,16,16 -> 256,8,8
        self.resblock_3 = ResBlock(4*self.d, 8*self.d, resample='down') #256,8,8 -> 512,4,4
        self.lrelu = nn.LeakyReLU(.2)
        self.linear = nn.Linear(self.d*8*4*4, 1)

    def forward(self, img):
        
        x = self.map(img) #3,32,32 -> 64,32,32
        x = self.resblock_1(x) 
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.lrelu(x)
        x = self.linear(x.reshape(-1, self.d*8*4*4))

        return x.squeeze() #[N,]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: #Conv&Transpose Conv
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: #Conv&Transpose Conv
        xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: #Conv&Transpose Conv
        kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)
