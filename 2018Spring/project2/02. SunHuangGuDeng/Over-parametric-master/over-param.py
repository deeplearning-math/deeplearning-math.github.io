# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import os
import sys
import pickle
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from cifar import CIFAR10
from models import resnet, alexnet, inceptions, vgg
from utils import AverageMeter, accuracy, Logger

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

use_gpu = torch.cuda.is_available()


def train_model(model, criterion, optimizer, scheduler, log_saver, mode, num_epochs=60):
    since = time.time()
    steps = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:

            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            for i, data in enumerate(loaders[phase]):
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                if phase == 'train':
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                else:
                    inputs = Variable(inputs, volatile=True)
                    labels = Variable(labels, volatile=True)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    steps += 1

                loss_meter.update(loss.data[0], outputs.size(0))
                acc_meter.update(accuracy(outputs.data, labels.data)[-1][0], outputs.size(0))

            epoch_loss = loss_meter.avg
            epoch_error = 1 - acc_meter.avg / 100

            if phase == 'train':
                log_saver.log_train(steps, epoch_loss, epoch_error)
            else:
                log_saver.log_test(steps, epoch_loss, epoch_error)

            print('{} Loss: {:.4f} Error: {:.4f}'.format(
                phase, epoch_loss, epoch_error))

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print('Saving..')
            state = {
                'net': model,
                'epoch': epoch,
                'log': log_saver
            }

            if not os.path.isdir('checkpoint_{}'.format(mode[2])):
                os.mkdir('checkpoint_{}'.format(mode[2]))
            torch.save(state, './checkpoint_{}/{}_{}_ckpt_epoch_{}.t7'.format(mode[2], mode[0], mode[1], epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, log_saver


## 'vgg16','resnet18','alex','inception'
model_name = 'alex'

root = './'
lr = 0.1
BATCH_SIZE = 128
weight_decay = 0.
num_epochs = 60

mode_set = [('normal', 'normal'), ('normal', 'random'), ('normal', 'partially-0.1'),
            ('normal', 'partially-0.3'), ('normal', 'partially-0.5'), ('normal', 'partially-0.7'),
            ('normal', 'partially-0.9'), ('random', 'normal'), ('shuffled', 'normal')]

for (mode1, mode2) in mode_set:
    print(mode1, ' and ', mode2, ' :')
    img_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(
                                             (0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010))])
    training_dataset = CIFAR10(root, train=True, transform=img_transforms, image_mode=mode1, label_mode=mode2)
    training_loader = DataLoader(training_dataset, BATCH_SIZE, shuffle=True, pin_memory=True)

    testing_dataset = CIFAR10(root, train=False, transform=img_transforms)
    testing_loader = DataLoader(testing_dataset, BATCH_SIZE, shuffle=False, pin_memory=True)

    loaders = {'train': training_loader, 'test': testing_loader}

    resnet18 = resnet.ResNet18()
    vgg16 = vgg.VGG('VGG16')
    alex = alexnet.alexnet()
    inception = inceptions.GoogLeNet()

    exec('model={}'.format(model_name))

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)

    mode = [mode1, mode2, model_name]
    log = Logger(mode)

    model, log = train_model(model, criterion, optimizer, exp_lr_scheduler, log, mode, num_epochs=num_epochs)


def plot(title):
    fontdict = {'size': 30}

    def get_fig(i):
        fig = plt.figure(i, figsize=(20, 10))
        ax = fig.add_subplot(111)
        plt.title(title, fontsize=40, y=1.04)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        return fig, ax

    fig1, ax1 = get_fig(1)
    fig2, ax2 = get_fig(2)
    fig3, ax3 = get_fig(3)
    fig4, ax4 = get_fig(4)

    for (mode1, mode2) in mode_set:
        state = torch.load('./checkpoint_{}/{}_{}_ckpt_epoch_{}.t7'.format(title, mode1, mode2, num_epochs - 1))
        log = state['log']
        label = mode1 + '-' + mode2
        ax1.plot(log.step_logger, log.loss_logger, linewidth=3, label=label)
        ax2.plot(log.step_logger, log.error_logger, linewidth=3, label=label)
        ax3.plot(log.step_logger_test, log.loss_logger_test, linewidth=3, label=label)
        ax4.plot(log.step_logger_test, log.error_logger_test, linewidth=3, label=label)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, log.step_logger[-1])
        ax.set_xlabel('steps', fontdict=fontdict)
        ax.legend(loc='upper right', fontsize=20)

    ax1.set_ylabel('train loss', fontdict=fontdict)
    ax2.set_ylabel('train error', fontdict=fontdict)
    ax3.set_ylabel('test loss', fontdict=fontdict)
    ax4.set_ylabel('test error', fontdict=fontdict)

    result_dir = './results-{}/'.format(title)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    fig1.savefig(result_dir + title + '-train-loss.png')
    fig2.savefig(result_dir + title + '-train-error.png')
    fig3.savefig(result_dir + title + '-test-loss.png')
    fig4.savefig(result_dir + title + '-test-error.png')


plot(model_name)
plt.show()
plt.close()
