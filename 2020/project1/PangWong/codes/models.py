import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import Adam
from torchvision import models


def create_model(opt):
    if opt.model == 'vgg11':
        return VGG11Model(opt)
    elif opt.model == 'resnet18':
        return Resnet18Model(opt)
    else:
        raise NotImplementedError


class TorchVisionModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.model = self.get_model(opt)

        self.use_gpu = (len(opt.gpu_ids) > 0)
        if self.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids).cuda()

        self.isTrain = opt.isTrain
        if opt.isTrain:
            self.model.train()
            self.loss = nn.CrossEntropyLoss()
        else:
            self.model.eval()

    def get_model(self, opt):
        raise NotImplementedError

    def preprocess_inputs(self, batch):
        if self.use_gpu:
            for k in batch.keys():
                batch[k] = batch[k].cuda()
        return batch

    def forward(self, batch, mode):
        batch = self.preprocess_inputs(batch)

        if mode == 'train':
            logits = self.model(batch['inputs'])
            loss = self.loss(logits, batch['labels'])
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch['labels']).float().mean()
            return logits, loss, acc

        elif mode == 'val':
            with torch.no_grad():
                batch_size = len(batch['labels'])
                logits = self.model(batch['inputs'])
                losses_sum = self.loss(logits, batch['labels']) * batch_size
                preds = torch.argmax(logits, dim=1)
                accs_sum = (preds == batch['labels']).float().sum()
            return losses_sum, accs_sum

        elif mode == 'eval':
            with torch.no_grad():
                logits = self.model(batch['inputs'])
            return logits

        else:
            raise NotImplementedError

    def create_optimizer(self, opt):
        params = list(self.model.parameters())
        optim = Adam(params, lr=opt.lr)
        return optim


class VGG11Model(TorchVisionModel):
    def get_model(self, opt):
        model = models.vgg11_bn(pretrained=True)
        seq = list(model.classifier.children())

        if opt.isTrain:
            features_dim = seq[-1].in_features
            head = nn.Linear(features_dim, 10)
            init.xavier_normal_(head.weight.data, 0.02)
            init.constant_(head.bias.data, 0.0)
            classifier = nn.Sequential(*seq[:-1], head)
        else:
            classifier = nn.Sequential(*seq[:-3])

        model.classifier = classifier
        return model


class Resnet18Model(TorchVisionModel):
    def get_model(self, opt):
        model = models.resnet18(pretrained=True)

        if opt.isTrain:
            features_dim = model.fc.in_features
            head = nn.Linear(features_dim, 10)
            init.xavier_normal_(head.weight.data, 0.02)
            init.constant_(head.bias.data, 0.0)
        else:
            head = nn.Identity()

        model.fc = head
        return model
