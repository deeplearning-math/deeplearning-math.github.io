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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from cifar import CIFAR10
from models import basic_cnn
from utils import AverageMeter, accuracy, Logger

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

use_gpu = torch.cuda.is_available()


def train_model(model, criterion, optimizer, log_saver, num_epochs=70):
    since = time.time()
    steps = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:

            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            if phase == 'train':
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

            if phase == 'train' and epoch == num_epochs - 1:

                log_saver['train_loss'].append(epoch_loss)
                log_saver['train_error'].append(epoch_error)

            elif phase == 'test' and epoch == num_epochs - 1:

                log_saver['test_loss'].append(epoch_loss)
                log_saver['test_error'].append(epoch_error)

            print('{} Loss: {:.4f} Error: {:.4f}'.format(
                phase, epoch_loss, epoch_error))

        if epoch % 30 == 0 or epoch == num_epochs - 1:
            print('Saving..')
            state = {
                'net': model,
                'epoch': epoch,
                'log': log_saver
            }

            if not os.path.isdir('checkpoint_CNN'):
                os.mkdir('checkpoint_CNN')
            torch.save(state, './checkpoint_CNN/ckpt_epoch_{}_{}.t7'.format(epoch, log_saver['num_params'][-1]))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, log_saver


model_name = 'CNN'

root = './'
lr = 0.01
BATCH_SIZE = 100
weight_decay = 0.
num_epochs = 70

img_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))])
training_dataset = CIFAR10(root, train=True, transform=img_transforms, image_mode='normal', label_mode='normal')
training_loader = DataLoader(training_dataset, BATCH_SIZE, shuffle=True, pin_memory=True)

testing_dataset = CIFAR10(root, train=False, transform=img_transforms)
testing_loader = DataLoader(testing_dataset, BATCH_SIZE, shuffle=False, pin_memory=True)

loaders = {'train': training_loader, 'test': testing_loader}

log = {'num_params': [],
       'train_loss': [],
       'train_error': [],
       'test_loss': [],
       'test_error': []}

num_channels = list(range(3, 120, 3)) + list(range(150, 450, 100))

# %% run the model
for channels in num_channels:
    print('Now at {}.............'.format(channels))
    model = basic_cnn.CNN(channels)

    number_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log['num_params'].append(number_params)

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model, log = train_model(model, criterion, optimizer, log, num_epochs=num_epochs)


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

    ax1.plot(np.log10(log['num_params']), log['train_loss'], linewidth=3, label='training')
    ax1.plot(np.log10(log['num_params']), log['test_loss'], linewidth=3, label='test')
    ax2.plot(np.log10(log['num_params']), log['train_error'], linewidth=3, label='training')
    ax2.plot(np.log10(log['num_params']), log['test_error'], linewidth=3, label='test')

    for ax in [ax1, ax2]:
        ax.set_xlim(2, 7)
        ax.set_xlabel('10^ Number of Params', fontdict=fontdict)
        ax.legend(loc='lower left', fontsize=20)

    ax1.set_ylabel('Loss on CIFAR10', fontdict=fontdict)
    ax2.set_ylabel('Error on CIFRA10', fontdict=fontdict)

    result_dir = './results-{}/'.format(title)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    fig1.savefig(result_dir + title + '-loss.png')
    fig2.savefig(result_dir + title + '-error.png')


plot(model_name)
plt.show()
plt.close()


# %% run the analysis
###################### Analysis
def get_prob(checkpoint_path):
    probs_param = np.zeros((len(num_channels), 10000, 10))
    checkpoints = [checkpoint for checkpoint in os.listdir(checkpoint_path)
                   if int(checkpoint.split('_')[2]) == num_epochs - 1]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for i, checkpoint in enumerate(checkpoints):
        model = torch.load(os.path.join(checkpoint_path, checkpoint))['net']
        step = 0
        for data in testing_loader:
            inputs, _ = data
            if use_gpu:
                inputs = inputs.cuda()
            inputs = Variable(inputs, volatile=True)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=-1).cpu().data.numpy()
            probs_param[i, step:(step + probs.shape[0]), :] = probs
            step += probs.shape[0]
    return probs_param


# get all the predictions and probs
probs_param = get_prob('./checkpoint_CNN')
prediction_param = np.stack([np.where(a.argmax(axis=1) == testing_dataset.test_labels, 1, 0) for a in probs_param])

# %% do some analysis
# in order to see how much percentage of data always predicted unchanged over every 10 models
d = 10
k = 0.7
a = [np.mean(prediction_param[i:i + d, :], axis=0) >= k for i in range(42 - d)]
a1 = [np.mean(prediction_param[i:i + d, :], axis=0) == 1 for i in range(42 - d)]
a2 = [np.mean(prediction_param[i:i + d, :], axis=0) == 0 for i in range(42 - d)]
a3 = [np.mean(prediction_param[i:i + d, :], axis=0) <= 1 - k for i in range(42 - d)]

plt.figure()
plt.plot(range(1, len(a) + 1), [np.mean(i) for i in a])
plt.title('Trend over every %d models' % d)
plt.xlabel('Start Model')
plt.ylabel('Percentage of 70% Remaining Correct')
plt.savefig('Trend over 70% of every {} models Correct'.format(d))
plt.show()

# plt.figure()
# plt.plot(range(1, len(a1) + 1), [np.mean(i) for i in a1])
# plt.title('Trend over every %d models' % d)
# plt.xlabel('Start Model')
# plt.ylabel('Percentage of Remaining Correct')
# plt.savefig('Trend over every %d models Correct' % d)
# plt.show()
# # in order to see how much percentage of data always predicted error over every 10 models
# plt.figure()
# plt.plot(range(1, len(a2) + 1), [np.mean(i) for i in a2])
# plt.title('Trend over every %d models' % d)
# plt.xlabel('Start Model')
# plt.ylabel('Percentage of Remaining Mistake')
# plt.savefig('Trend over every %d models Mistake' % d)
# plt.show()

plt.figure()
plt.plot(range(1, len(a3) + 1), [np.mean(i) for i in a3])
plt.title('Trend over every %d models' % d)
plt.xlabel('Start Model')
plt.ylabel('Percentage of 70% Remaining Mistake')
plt.savefig('Trend over 70% of every %d models Mistake'.format(d))
plt.show()
