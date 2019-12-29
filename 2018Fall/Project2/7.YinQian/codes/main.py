import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
import time

from vgg_dcf import vgg16_bn as dcf_vgg16bn
from resnet_dcf import resnet18 as dcf_resnet18

from torchvision.models import vgg16_bn as cnn_vgg16bn
from torchvision.models import resnet18 as cnn_resnet18


parser = argparse.ArgumentParser(description='DCFNet')

parser.add_argument('--model', type=str, help='Model to use')
parser.add_argument('--num_bases', type=int, default=6,
                    help='Number of DCF bases to use')

parser.add_argument('--max_epoch', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size')

parser.add_argument('--optim', type=str, default='SGD',
                    help='optimization algorithm')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='CUDA')

parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--log_interval', type=int, default=10,
                    help='report interval')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed(args.seed)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_sz = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_sz))
        return res


def evaluate(data_source, model, criterion):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_source):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels).to(args.device)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

        return losses.avg, top1.avg, top5.avg


def train(data_source, model, optimizer, criterion, epoch):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (inputs, labels) in enumerate(data_source):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % args.log_interval == 0 and idx > 0:
            print('| Epoch: {} | {}/{} Batches '
                  '| Loss {:.3f} | acc1 {:.3f} | acc5 {:.3f} '.format(epoch, idx, len(data_source),
                                                                      losses.avg, top1.avg, top5.avg))

    return losses.avg, top1.avg, top5.avg


def main(args):
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    writer = SummaryWriter(log_dir=args.save)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(32, 4),
                                         transforms.ToTensor(), normalize])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(), normalize])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0)

    # print('Constructing model')
    # model = vgg11_bn(num_classes=10).to(args.device)

    if args.model == 'cnn_vgg16bn':
        model = cnn_vgg16bn(num_classes=10).to(args.device)
    elif args.model == 'cnn_resnet18':
        model = cnn_resnet18(num_classes=10).to(args.device)
        model.fc = nn.Sequential(
            nn.Dropout(),
            model.fc
        )
        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize([224, 224]),
                                             transforms.RandomHorizontalFlip(),
                                             # transforms.RandomCrop(32, 4),
                                             transforms.ToTensor(), normalize])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=0)

        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize([224, 224]),
                                             transforms.ToTensor(), normalize])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=0)
    elif args.model == 'dcf_vgg16bn':
        model = dcf_vgg16bn(num_bases=args.num_bases, num_classes=10).to(args.device)
    elif args.model == 'dcf_resnet18':
        model = dcf_resnet18(num_bases=args.num_bases, num_classes=10).to(args.device)
        model.fc = nn.Sequential(
            nn.Dropout(),
            model.fc
        )

    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(x.data.nelement() for x in model.parameters())
    print('Model parameters: {}'.format(total_params))

    criterion = nn.CrossEntropyLoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=args.wd)

    best_acc1 = 0

    try:
        for epoch in range(1, args.max_epoch + 1):

            lr = args.lr * (0.1 ** (epoch // 30))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            print('Start training')
            epoch_start_time = time.time()
            train_loss, train_acc1, train_acc5 = train(trainloader, model, optimizer, criterion, epoch)

            print('| Epoch: {} | Time: {:.3f}s '
                  '| Loss {:.3f} | | acc1 {:.3f} | acc5 {:.3f} '.format(epoch, time.time() - epoch_start_time,
                                                                        train_loss, train_acc1, train_acc5))
            print('-' * 100)

            writer.add_scalar('Training/Loss', train_loss, epoch)
            writer.add_scalar('Training/Top1_acc', train_acc1, epoch)
            writer.add_scalar('Training/Top5_acc', train_acc5, epoch)

            for name, param in model.named_parameters():
                writer.add_histogram(name, param.data.cpu().numpy(), epoch)
                writer.add_histogram(name + '/grad', param.grad.cpu().numpy(), epoch)

            print('Start testing')
            eval_loss, eval_acc1, eval_acc5 = evaluate(testloader, model, criterion)
            print('| End of Epoch: {} | Time: {:.3f}s '
                  '| Loss {:.3f} | | acc1 {:.3f} | acc5 {:.3f} '.format(epoch, time.time() - epoch_start_time,
                                                                        eval_loss, eval_acc1, eval_acc5))
            print('-' * 100)

            writer.add_scalar('Test/Loss', eval_loss, epoch)
            writer.add_scalar('Test/Top1_acc', eval_acc1, epoch)
            writer.add_scalar('Test/Top5_acc', eval_acc5, epoch)

            if eval_acc1 > best_acc1:
                print('Saving model')
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc1_te': eval_acc1
                }
                torch.save(state, '{}.pt'.format(args.save))
                best_acc1 = max(eval_acc1, best_acc1)

    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')


if __name__ == '__main__':
    main(args)
