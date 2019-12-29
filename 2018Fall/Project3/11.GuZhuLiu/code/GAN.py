import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import time
from data import *
from config import Config as conf
import matplotlib.pyplot as plt
from model import generator, discriminator
import torch.optim as optim
from padding_same import Conv2d_same_padding

# Set Hyperparameters

batch_size = 2
learning_rate = 0.001
size = 10000 - 64
d = 64
test_number = 1
scale = 1
Bsize = 10000
lrg = 0.01
lrd = 0.001
n = 5

# Download Data
USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


####preprocess step
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


####network
# network
G = generator(d)
D = discriminator(d)

# Binary Cross Entropy loss
BCE_loss = nn.BCEWithLogitsLoss()
D_losses = []
G_losses = []

##data
data = load_data()

if not os.path.isdir():
    os.mkdir('./save_net/gan/model' + str(n))
if not os.path.isdir('./save_net/gan/model' + str(n)):
    os.mkdir('./loss/dloss' + str(n))
if not os.path.isdir('./loss/dloss' + str(n)):
    os.mkdir('./loss/gloss' + str(n))

###train

for epoch in range(conf.num_epochs):
    k = 0
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)

    G.cuda()

    D.cuda()
    G_optimizer = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.999))
    train_data = data["train"]()
    epoch_start_time = time.time()
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
            y_real_ = torch.ones((batch_size, 512, 26, 26))
            y_fake_ = torch.zeros((batch_size, 512, 26, 26))
            y_real_ = y_real_.to(device=device, dtype=dtype)
            y_fake_ = y_fake_.to(device=device, dtype=dtype)
            ###d_step
            D.zero_grad()
            D_result = D(pimg2, pcond2)
            D_result = D_result.squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)
            G_result = G(pcond2)
            D_result = D(G_result, pcond2).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            # D_losses.append(D_train_loss.data[0])
            D_losses.append(D_train_loss.item())

            #  G_step

            G.zero_grad()

            G_result = G(pcond2)
            D_result = D(G_result, pcond2).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_) + conf.L1_lambda1 * (torch.mean(torch.abs(G_result - pimg2)))
            G_train_loss.backward(retain_graph=True)
            G_optimizer.step()

            G_losses.append(G_train_loss.item())
            pimg = []
            pcond = []

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        if k % 10 == 0:
            print('[%d/%d] - lrd:%.5f, lrg:%.5f, lgptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
                k, epoch, lrd, lrg, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                torch.mean(torch.FloatTensor(G_losses))))
    torch.save(G.state_dict(),
               'D:\cryo_em\code\denoise_code\pytorch_gan\save_net\gan\\model' + str(n) + '\\G_%d' % (epoch))
    torch.save(D.state_dict(),
               'D:\cryo_em\code\denoise_code\pytorch_gan\save_net\gan\\model' + str(n) + '\\D_%d' % (epoch))
    np.save('D:\cryo_em\code\denoise_code\pytorch_gan\loss\\dloss2' + str(n) + '\\' + 'dloss_%d' % (epoch + 1),
            D_losses)
    np.save('D:\cryo_em\code\denoise_code\pytorch_gan\loss\\gloss2' + str(n) + '\\' + 'gloss_%d' % (epoch + 1),
            G_losses)
