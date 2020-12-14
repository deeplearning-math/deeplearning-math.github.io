import os
import argparse
import pandas as pd
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, NLLLoss, BCELoss
from torchvision import models

from data import get_loaders
from model import ImageBinaryModel

def train(model, train_loader, valid_loader, criterion, optimizer, epochs):
    best_valid_loss = 100000
    for epoch in range(1, epochs+1):
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc=f'Training', total=len(train_loader), ncols=100):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            # print("ouput", output.dtype)
            # print("target", target.dtype)
            # print("target", target, target.float(), target.float().dtype)

            loss = criterion(output, target.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
        
        model.eval()
        valid_loss = 0
        correct = 0
        total_samples = 0
        for data, target in tqdm(valid_loader, desc=f'Validation', total=len(valid_loader), ncols=100):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            valid_loss += criterion(output, target.float().unsqueeze(1)).item()
            # print(output.data)
            # print(output.data.ge(0.5))
            pred = output.data.ge(0.5)

            correct += pred.eq(target.unsqueeze(1)).cpu().sum()
            # print(pred.eq(target.unsqueeze(1)))
            # input()
            total_samples += pred.shape[0]

        valid_loss /= len(valid_loader) # loss function already averages over batch size
        print('Valid Epoch: {} \tLoss: {:.6f} \tCorrect {}/{} {:.6f}'.format(epoch, valid_loss, correct, total_samples, float(correct)/float(total_samples)))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
    
    return model


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    for data, _ in test_loader:
        data = data.cuda()
        output = model(data)
        preds.extend(output.cpu().detach().numpy().tolist())
    return preds

if __name__ == "__main__":
    lr = 0.01
    momentum = 0.5
    epochs = 3
    batch_size = 20
    test_batch_size = 20

    # set random seed
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # prepare data
    train_path = r"semi-conductor-image-classification-second-stage/train/train_contest"
    test_path = r"semi-conductor-image-classification-second-stage/test/test_contest"
    train_loader, valid_loader, test_loader = get_loaders(train_path, test_path, batch_size, test_batch_size)

    # prepare model
    model = ImageBinaryModel(base_model="vgg16", freeze_base=True)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.BCELoss()

    # train
    model = train(model, train_loader, valid_loader, criterion, optimizer, epochs)
    # save best model

    # inference
    preds = test(model, test_loader)
    print("preds", preds)

    # save results

