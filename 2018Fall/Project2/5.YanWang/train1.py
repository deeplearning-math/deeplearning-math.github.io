import torch
import torch.nn as nn
import numpy as np
import time

from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import resnet18
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.metrics import accuracy_score,roc_auc_score
import os
import nibabel as nib
import argparse
from utils import AverageMeter
from distutils.version import LooseVersion
import math
# from losses import DICELossMultiClass

def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()

    model.train()
    for iteration, (sample,target) in enumerate(train_loader):

        image = Variable(sample).float().cuda()
        label = Variable(target).long()
        out = model(image)
        out = torch.clamp(out, min = 0.0, max = 1.0)
        out = out.contiguous().view(-1, args.num_classes)
        label = label.contiguous().view(-1).cuda()
        # print(out.size(), label.size())
        loss = criterion(out, label)
        losses.update(loss.data[0],image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust learning rate
        cur_iter = iteration + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizer, cur_iter, args)

        if iteration%200==0:
            print('   * i {} |  lr: {:.7f} | Training Loss: {losses.avg:.3f}'.format(iteration, args.running_lr, losses=losses))
    print('   * EPOCH {epoch} | Training Loss: {losses.avg:.3f}'.format(epoch=epoch, losses=losses))


def validation(val_loader, model,epoch, args):
    model.eval()
    label_list = []
    prob_list = []
    for iteration, (sample,target) in enumerate(val_loader):
        image = Variable(sample).float().cuda()
        label = Variable(target).long().data.cpu().numpy()
        out = model(image)
        with torch.no_grad():
            out_label = torch.max(out, 1)[1].cpu().numpy()

            out = out.data.cpu().numpy()
            # print(out,label)
            for j in range(out.shape[0]):
                # name_list.append(name[j])
                label_list.append(label[j])
                # prob = out[j,1]/(out_label[j,0]+out_label[j,1])
                prob_list.append(out[j,1])
                # prob_list.append(out_label[j])
    # label = np.array(label_list)
    # prob = np.array(prob_list)
    #
    # accu1 = float(np.sum(label))/flaot(len(label))
    # accu1 = accuracy_score(label[label==1],prob[prob==1])
    # accu = accuracy_score(label[label==0],prob[prob==0])
    accu = roc_auc_score(np.array(label_list),np.array(prob_list))
    print('   * EPOCH {epoch} | Validation AUC: {losses} '.format(epoch=epoch, losses=accu))
    return accu        # print(out[0,0])
            # break

def save_checkpoint(state, epoch, args):
    filename = args.ckpt + '/' + str(epoch) + '_checkpoint.pth.tar'
    print(filename)
    torch.save(state, filename)

# def load_checkpoint(epoch, args):

    # torch.save(state, filename)


def adjust_learning_rate(optimizer, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr

def main(args):
    # import network architecture
    # builder = ModelBuilder()
    model = resnet18(num_classes = 2)
    # model = resnet18(num_classes = 2)
    #model = models.vgg11_bn( num_classes = 2)
    # model = models.AlexNet( num_classes = 2)
    # model = models.densenet121(num_classes = 2)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
    cudnn.benchmark = True

    # collect the number of parameters in the network
    print("------------------------------------------")
    # print("Network Architecture of Model %s:" % (args.id))
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul

    print(model)
    print("Number of trainable parameters %d in Model" % (num_para))
    print("------------------------------------------")

    # set the optimizer and loss
    #optimizer = optim.RMSprop(model.parameters(), args.lr, alpha=args.alpha, eps=args.eps, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), args.lr, eps=args.eps, weight_decay=args.weight_decay)
    # optimizer = optim.SparseAdam(model.parameters(), args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)


    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight = torch.cuda.FloatTensor([1.0,9.0]))
    # criterion = DICELossMultiClass()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    # loading data
    data_transforms = transforms.Compose([
        # transforms.CenterCrop(192),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(degrees = args.rotate_range),
        transforms.ToTensor(),
        transforms.Normalize([63.321, 63.321, 63.321], [40.964, 40.964, 40.964])
    ])
    dataset = datasets.ImageFolder(root=args.root_path, transform = data_transforms )
    num_train = len(dataset)
    # print(num_train)
    indices = list(range(num_train))
    split = num_train/10
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    # tf = TrainDataset(train_dir, args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,sampler=validation_sampler)

    print("Start training ...")
    best_epoch=0
    best_accu= 0
    for epoch in range(args.start_epoch + 1, args.num_epochs + 1):
        start_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args)

        # save models
        # if (epoch >= args.particular_epoch and epoch % args.save_epochs_steps == 0) or epoch % args.save_epochs_steps_before == 0 :
            # if epoch % args.save_epochs_steps == 0:
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, epoch, args)
        filename = args.ckpt + '/' + str(epoch) + '_checkpoint.pth.tar'
        # print(filename)
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        accu = validation(validation_loader, model, epoch, args)
        if accu > best_accu:
            best_epoch = epoch
            best_accu = accu
        elapsed_time = time.time() - start_time
        print('epoch time ' +str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + ', remaining time ' +str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time*(args.num_epochs - epoch)))))

    print("Training Done")
    print("Best epoch " +str(best_epoch) + ', auc score '+str(best_accu))

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments


    # Path related arguments

    parser.add_argument('--root_path', default='/home/wynonna/Documents/Research/Brats2018/course_pro/data/train/',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output ciheckpoints')
    parser.add_argument('--num_round', default=21, type=int)

    # Data related arguments

    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of data loading workers')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='if shuffle the data during training')
    parser.add_argument('--rotate_range', default=5, type=int,
                        help='if shuffle the data during training')

    # optimization related arguments
    parser.add_argument('--random_sample', action='store_true', default=True, help='whether to sample the dataset with random sampler')
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=48, type=int,
                        help='training batch size')
    parser.add_argument('--num_epochs', default=60, type=int,
                        help='epochs for training')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--lr', default=8e-5, type=float,
                        help='start learning rate')
    parser.add_argument('--lr_pow', default=0.98, type=float,
                        help='power in poly to drop learning rate')
    parser.add_argument('--optim', default='RMSprop', help='optimizer')
    parser.add_argument('--alpha', default='0.9', type=float, help='alpha in RMSprop')
    # parser.add_argument('--betas', default='(0.9,0.999)', type=float, help='betas in Adam')
    parser.add_argument('--eps', default=10**(-4), type=float, help='eps in RMSprop')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='weights regularizer')
    parser.add_argument('--momentum', default=0.8, type=float, help='momentum for RMSprop')
    parser.add_argument('--save_epochs_steps', default=1, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--save_epochs_steps_before', default=10, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--particular_epoch', default=40, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')


    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # train_file = open(args.train_path, 'r')
    # train_dir = train_file.readlines()

    args.ckpt = os.path.join(args.ckpt, str(args.num_round))
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_checkpoint.pth.tar'

    args.running_lr = args.lr
    args.epoch_iters = math.ceil(int(30000/args.batch_size))
    args.max_iters = args.epoch_iters * args.num_epochs


    main(args)
