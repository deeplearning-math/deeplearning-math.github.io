import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models import resnet18
import torchvision.models as models
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

from dataset import TestDataset
import os
import nibabel as nib
import argparse
from utils import AverageMeter
from distutils.version import LooseVersion
import math
# from losses import DICELossMultiClass

def train(train_loader, model, args):
    # losses = AverageMeter()

    model.eval()
    name_list = []
    label_list = []
    prob_list = []
    for iteration, sample in enumerate(train_loader):

        image = Variable(sample['images']).float().cuda()
        name = sample['names'] # no cuda for make one hot !!!

        # The dimension of out should be in the dimension of B,C,W,H,D
        # transform the prediction and label
        out = model(image)
        out = torch.clamp(out, min = 0.0, max = 1.0)
        with torch.no_grad():
            out_label = torch.max(out, 1)[1].cpu().numpy()
            out = out.data.cpu().numpy()
            for j in range(out.shape[0]):
                name_list.append(name[j])
                label_list.append(out_label[j])
                # prob = out[j,1]/(out_label[j,0]+out_label[j,1])
                prob_list.append(out[j,1]/(out[j,0]+out[j,1]))
            # print(out)
            # break
    raw_data={'id':name_list, 'label':prob_list}
    df = pd.DataFrame(raw_data,columns = ['id','label'])
    df.to_csv('eval-' +str(args.num_round)+'-'+str(args.epoch_index)+'.csv',index= False)
def main(args):

    model = resnet18(num_classes = 2)
    # model = models.densenet121(num_classes = 2)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
    cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            raise Exception("=> No checkpoint found at '{}'".format(args.resume))

    tf = TestDataset(val_dir, args)
    num_images = len(val_dir)

    # dataset = datasets.ImageFolder(root=args.root_path, transform = data_transforms )
    # tf = TrainDataset(train_dir, args)
    train_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=True)

    print("Start testing ...")


    train(train_loader, model, args)
        # elapsed_time = time.time() - start_time
        # print('epoch time ' +str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + ', remaining time ' +str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time*(args.num_epochs - epoch)))))

        # save models
        # if (epoch >= args.particular_epoch and epoch % args.save_epochs_steps == 0) or epoch % args.save_epochs_steps_before == 0 :
            # if epoch % args.save_epochs_steps == 0:
            # save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, epoch, args)

    print("Testing Done")

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments


    # Path related arguments
    parser.add_argument('--val_path', default='/home/wynonna/Documents/Research/Brats2018/course_pro/test_list.txt',
                        help='txt file of the name of validation data')
    parser.add_argument('--root_path', default='/home/wynonna/Documents/Research/Brats2018/course_pro/data/all_tests/',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output ciheckpoints')


    # Data related arguments
    parser.add_argument('--center_crop', default=True, type=bool,
                        help='crop from center or just random')
    # parser.add_argument('--center_crop_size', default=[160,200,144] [41,201,21,221,7,151], nargs='+', type=int,
    #                     help='crop size of the input image (int or list)')[152,192,xx][45,197,25,117,xxxx]

    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='number of data loading workers')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='if shuffle the data during training')
    parser.add_argument('--rotate_range', default=5, type=int,
                        help='if shuffle the data during training')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='validation batch size')
    parser.add_argument('--num_epochs', default=1, type=int,
                        help='epochs for validation')
    parser.add_argument('--start_epoch', default=40, type=int,
                        help='epoch to start validation.')
    parser.add_argument('--steps_spoch', default=10, type=int,
                        help='number of epochs to do the validation')
    parser.add_argument('--best_epoch', default=0, type=int,
                        help='number of epochs has the best validation result')
    parser.add_argument('--best_mean', default=0.0, type=float,
                        help='the best mean dice score')
    parser.add_argument('--num_round', default=20, type=int,
                        help='restore the models from which run')
    parser.add_argument('--visualize', default = False, type = bool, #action='store_true',
                        help='save the prediction result as 3D images')
    parser.add_argument('--result', default='./result',
                        help='folder to output prediction results')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    val_file = open(args.val_path, 'r')
    val_dir = val_file.readlines()

    args.ckpt = os.path.join(args.ckpt, str(args.num_round))
    # print('Models are saved at %s' % (args.ckpt))

    for i in range(args.num_epochs):
        args.epoch_index = args.start_epoch + args.steps_spoch * i
        args.resume = args.ckpt + '/' + str(args.epoch_index) + '_checkpoint.pth.tar'
        main(args)

    print('Best Epoch: %d, Best mean epoch: %.4f' % (args.best_epoch, args.best_mean))
