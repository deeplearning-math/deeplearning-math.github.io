# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import os
import re
import random
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms, models

num_crops = 200
crop_size = 224
use_cuda = torch.cuda.is_available()

root_dir = './'
path1 = './cropped_images/'
path2 = './cropped_feats/'

if not os.path.exists(path1):
    os.makedirs(path1)
if not os.path.exists(path2):
    os.makedirs(path2)

names = [name for name in os.listdir(root_dir) if name.lower().endswith(('tif', 'tiff', 'jpg'))]
names.sort(key=lambda x: int(re.findall(r'(\d+).', x)[0]))
labels = [0] + [1] * 5 + [0] + [1] * 2 + [0] + [-1] * 9 + [0] + [1] * 2 + [0] + [1] + [0] * 2 + [1] * 2

transformations = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

pretrained_model = models.resnet18(pretrained=True)
num_feat = pretrained_model.fc.in_features
modules = list(pretrained_model.children())[:-1]
resnet18_feature = torch.nn.Sequential(*modules)
if use_cuda:
    resnet18_feature = resnet18_feature.cuda()

resnet18_feature.eval()

###### extract features random
for i, name in enumerate(names):
    img = Image.open(os.path.join(root_dir, name))
    w, h = img.size
    all_imgs = []
    for j in range(num_crops):
        random.seed(j)
        w1 = random.uniform(0, w - crop_size)
        np.random.seed(j)
        h1 = np.random.uniform(0, h - crop_size)
        img1 = img.crop((w1, h1, w1 + crop_size, h1 + crop_size))
        img1 = img1.resize((224, 224))
        img1.save(path1 + '%d_%d_%s' % (i + 1, j + 1, name))
        img1 = img1.convert('RGB')
        img1 = transformations(img1)
        all_imgs.append(img1)
    all_imgs = torch.stack(all_imgs)
    if use_cuda:
        all_imgs = all_imgs.cuda()
    all_imgs = Variable(all_imgs, volatile=True)
    features = resnet18_feature.forward(all_imgs).cpu().squeeze().data.numpy()
    features = np.hstack((features, np.array([labels[i]] * num_crops)[:, None]))
    np.save(path2 + name + '.features', features)

####### grid crop image
# scale = 4
# for i, name in enumerate(names):
#     img = Image.open(os.path.join(root_dir, name))
#     w, h = img.size
#     w0, h0 = w // scale, h // scale
#     all_imgs = []
#     for j in range(scale):
#         for k in range(scale):
#             w1 = j * w0
#             h1 = k * h0
#             img1 = img.crop((w1, h1, w1 + w0, h1 + h0))
#             img1 = img1.resize((224, 224))
#             img1.save(path1 + '%d_%d_%d_%s' % (i + 1, j + 1, k + 1, name))
#             img1 = img1.convert('RGB')
#             img1 = transformations(img1)
#             all_imgs.append(img1)
#     all_imgs = torch.stack(all_imgs)
#     if use_cuda:
#         all_imgs = all_imgs.cuda()
#     all_imgs = Variable(all_imgs, volatile=True)
#     features = resnet18_feature.forward(all_imgs).cpu().squeeze().data.numpy()
#     features = np.hstack((features, np.array([labels[i]] * scale**2)[:, None]))
#     np.save(path2 + name + '.features', features)


