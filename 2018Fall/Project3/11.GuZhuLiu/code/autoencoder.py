# Simple Convolutional Autoencoder
# Code by GunhoChoi

import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from data import *
from config import Config as conf
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model import generator

# Set Hyperparameters

epoch = 50
batch_size = 10
learning_rate = 0.001
size = 10000 - 64
d = 64
test_number = 1
scale = 1
Bsize = 10000
n = 3 ### model
# Download Data
USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Set Data Loader(input pipeline)


# Encoder
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 1 x 28 x 28 -> batch x 512

# Encoder
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 1 x 28 x 28 -> batch x 512

G = generator(d).cuda()


def prepocess_train(img, cond, ):
    # img = scipy.misc.imresize(img, [conf.adjust_size, conf.adjust_size])
    # cond = scipy.misc.imresize(cond, [conf.adjust_size, conf.adjust_size])
    # h1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    # w1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    # img = img[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    # cond = cond[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    if np.random.random() > 0.5:
        img = np.fliplr(img)
        cond = np.fliplr(cond)
    img = img / 127.5 - 1.
    cond = cond / 127.5 - 1.
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    return img, cond


def to_img(x):
    x = 127.5 * (x + 1)
    x = x.view(x.size(0), 1, 256, 256)
    return x


##union image
def union_image(image, scale):
    union = np.zeros((256, 256))
    for k in range(np.power(scale, 2)):
        i = k // scale
        j = k % scale
        union[d * 2 * j:d * 2 * j + d * 2, d * 2 * i:d * 2 * i + d * 2] = image[k]
    union = np.array(union)
    plt.imshow(union, cmap='gray')
    plt.show()

    return


loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, weight_decay=1e-5)

# train encoder and decoder
print('traning start')

data = load_data()
G.weight_init(mean=0.0, std=0.02)

for epoch in range(conf.num_epochs):
    k = 0
    train_data = data["train"]()
    pimg = []
    pcond = []
    for img, cond, name in train_data:
        pimg1, pcond1 = prepocess_train(img, cond)
        pimg.append(pimg1)
        pcond.append(pcond1)
        k = k + 1
        if (k % batch_size == 0):
            pimg2 = np.array(pimg)
            # print(pimg2.shape)
            pcond2 = np.array(pcond)
            pimg2 = np.reshape(pimg2, (batch_size, 1, 256, 256))
            pcond2 = np.reshape(pcond2, (batch_size, 1, 256, 256))
            pimg2 = torch.from_numpy(pimg2)
            pcond2 = torch.from_numpy(pcond2)
            pimg2 = pimg2.to(device=device, dtype=dtype)
            pcond2 = pcond2.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            output = G(pcond2)
            loss = loss_func(output, pimg2)
            loss.backward()
            optimizer.step()
            pimg = []
            pcond = []

        if k % 1500 == 0:
            print(k)
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, conf.num_epochs, loss.data[0]))
    if epoch % 1 == 0:

        torch.save(G.state_dict(),
                   'D:\cryo_em\code\denoise_code\pytorch_gan\save_net\\autoencoder\\model' + str(n) + '\\G_%d' % (epoch + 1))

####
