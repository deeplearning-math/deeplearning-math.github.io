#### CIFAR_10 ResNet GB-GAN

## Main changes: 1.optimize input pool
##               2.loss func built-in

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd

from network import Discriminator_Res, Generator, weights_init_normal, weights_init_xavier, weights_init_kaiming
from models.resnet import * 
from utils import img_truncate, PoolSet, inception_score
#from util import plot_scatter, plot_scatter_label, weights_init, compute_acc

parser = argparse.ArgumentParser()
parser.add_argument('--z_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--num_epoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--G_batch_size', type=int, default=128) #?
parser.add_argument('--pool_size_t', type=int, default=10)

parser.add_argument('--lr_d', type=float, default=0.001, help='learning rate discriminator, default=0.0002')
parser.add_argument('--eta', type=float, default=.01, help='Update rate for each period') #?default resonable?
parser.add_argument('--lr_g', type=float, default=0.01, help='learning rate generator, default=0.0002')
parser.add_argument('--dsteps', type=int, default=1, help='Num of steps of Discriminator for one step Generator')
parser.add_argument('--G_optimizer', type=str, default='Adam', help='Adam od SGD')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--init', type=str, default='xavier', help='xavier, kaiming or normal')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--period', type=int, default=20)
parser.add_argument('--G_epoch', type=int, default=10, help='number of epochs to train for') #default?

parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--disc', type=str, default='Res', help='DC or Res')
parser.add_argument('--exp', type=str, default='001')
#parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

# specify the gpu id if using only 1 gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    os.mkdir('./exp')
except:
    pass

try:
    os.mkdir('./exp/' + opt.exp)
    os.mkdir('./exp/' + opt.exp + '/results')
    os.mkdir('./exp/' + opt.exp + '/results/fix')
    os.mkdir('./exp/' + opt.exp + '/results/random')
except:
    pass


try:
    os.mkdir('./exp/' + opt.exp + '/results/loss_monitor')
except:
    pass

with open('./exp/' + opt.exp + '/loss.txt', 'a') as inFile:
    print(opt, file=inFile)

# dataset CIFAR10
dataset = datasets.CIFAR10(
    root='./data', download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # --> [-1, 1]
    ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True)


if opt.init == 'xavier':
    weights_init_ = weights_init_xavier
elif opt.init == 'kaiming':
    weights_init_ = weights_init_kaiming
elif opt.init == 'normal':
    weights_init_ = weights_init_normal
else:
    raise NameError('Wrong Initialization')

netG = Generator(z_dim=opt.z_dim)
netG.apply(weights_init_)
netG.to(device)

if opt.disc == 'Res':
    netD = Discriminator_Res()
elif opt.disc == 'DC':
    netD = Discriminator_DC()
else:
    raise NameError('Wrong Discriminator')
netD.apply(weights_init_)
netD.to(device)

## Resuming Models
if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    state = torch.load('./checkpoint/cifar_exp_' + opt.exp +'.t7')
    netG.load_state_dict(state['netG'])
    netD.load_state_dict(state['netD'])
    start_epoch = state['epoch'] + 1
    is_score = state['is_score']
    best_is = state['best_is']
    loss_G = state['loss_G']
    loss_D = state['loss_D']
    grad_G = state['grad_G']

else:
    start_epoch = 0
    is_score = []
    best_is = 0.0
    loss_G = []
    loss_D = []
    grad_G = []

netIncept = ResNet18()
netIncept.to(device)
netIncept = torch.nn.DataParallel(netIncept)
if not torch.cuda.is_available():
    checkpoint = torch.load('./checkpoint/res18_ckpt_cifar10.t7', map_location=lambda storage, loc: storage)
    netIncept.load_state_dict(checkpoint['net'])
    #net_feat.load_state_dict(checkpoint['net'])

else:
    checkpoint = torch.load('./checkpoint/res18_ckpt_cifar10.t7')
    netIncept.load_state_dict(checkpoint['net'])
    #net_feat.load_state_dict(checkpoint['net'])

print('-----------NetLoad Finished')

pool_size = opt.batch_size * opt.pool_size_t

## training placeholder
z_b = torch.FloatTensor(opt.batch_size, opt.z_dim).to(device)
img_b = torch.FloatTensor(opt.batch_size, 3, 32, 32).to(device)
p_z = torch.FloatTensor(pool_size, opt.z_dim).to(device)
p_img = torch.FloatTensor(pool_size, 3, 32, 32).to(device)
d_b = torch.FloatTensor(opt.batch_size).to(device) # disc label

## evaluation placeholder
show_z_b = torch.FloatTensor(64, opt.z_dim).to(device)
eval_z_b = torch.FloatTensor(250, opt.z_dim).to(device) # 250/batch * 120 --> 300000

## fix evaluation place holder
fix_show_z = torch.FloatTensor(64, opt.z_dim).to(device)
fix_show_z.normal_()

optim_D = optim.RMSprop(netD.parameters(), lr=opt.lr_d) # other param?
optim_G = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999)) #?suitable
#scheduler_D = optim.lr_scheduler.StepLR(optim_D, step_size=200, gamma=.1)
#scheduler_G = optim.lr_scheduler.StepLR(optim_G, step_size=200, gamma=.1)

criterion_G = nn.MSELoss()
criterion_D = nn.BCEWithLogitsLoss() ## Sigmoid + BCE, input is score

for epoch in range(start_epoch, start_epoch + opt.num_epoch):
    #print('Start epoch: %d' % epoch)
    ## input_pool: [pool_size, opt.z_dim] -> [pool_size, 32, 32]
    #scheduler_G.step()
    #scheduler_D.step()

    netD.train()
    netG.eval()
    p_z.normal_()
    p_img.copy_(netG(p_z).detach())

    loss_D_t = []
    loss_G_t = []
    grad_G_t = []

    for t in range(opt.period): 

        for _ in range(opt.dsteps):
            
            t = time.time()
            ### Update D
            netD.zero_grad()
            ## real
            real_img, _ = next(iter(dataloader)) # [batch_size, 1, 32, 32]
            img_b.copy_(real_img.to(device))
            real_D_err = criterion_D(netD(img_b), d_b.fill_(1))
            #real_D_err = torch.log(1 + torch.exp(-netD(img_b))).mean()
            real_D_err.backward()

            ## fake
            z_b_idx = random.sample(range(pool_size), opt.batch_size)
            img_b.copy_(p_img[z_b_idx])
            fake_D_err = criterion_D(netD(img_b), d_b.fill_(0))
            #fake_D_err = torch.log(1 + torch.exp(netD(img_b))).mean() # torch scalar[]
            fake_D_err.backward()

            loss_D_t.append((real_D_err+fake_D_err).detach().cpu().item())
            #print(loss_D_t[-1])
            #print((real_D_err+fake_D_err).detach().cpu().item())
            optim_D.step()

        ## update input pool            
        p_img_t = p_img.clone().to(device) ##!!DOUBLE SIZE OF INPUT POOL
        p_img_t.requires_grad_(True)
        if p_img_t.grad is not None:
            p_img_t.grad.zero_()
        fake_D_score = netD(p_img_t)
        fake_D_err = criterion_D(fake_D_score, torch.ones(pool_size).to(device)).detach().cpu().item()

        #fake_D_score.backward(torch.ones(pool_size).to(device))
        #p_update = p_img_t.grad * opt.eta
        dsdimg = autograd.grad(outputs=fake_D_score, inputs=p_img_t, 
                      grad_outputs=torch.ones(pool_size).to(device),
                      retain_graph=False)[0]
        p_update = dsdimg * opt.eta
        p_img = img_truncate(p_img + p_update)

        loss_G_t.append(fake_D_err)
        grad_G_t.append(np.abs(p_update.detach().cpu().numpy()).mean())

    ##update G after several steps
    netG.train()
    netD.eval()
    poolset = PoolSet(p_z.cpu(), p_img.cpu())
    poolloader = torch.utils.data.DataLoader(poolset, batch_size=opt.G_batch_size, shuffle=True)

    ## record epoch loss
    loss_G.append(np.mean(loss_G_t))
    loss_D.append(np.mean(loss_D_t))
    grad_G.append(np.mean(grad_G_t)) # ave update / ave #pixels larger than .5

    for _ in range(opt.G_epoch):

        approx_G_t = []
        for _, data_ in enumerate(poolloader, 0):
            netG.zero_grad()

            input_, target_ = data_
            pred_ = netG(input_.to(device))
            loss = criterion_G(pred_, target_.to(device))
            loss.backward()

            optim_G.step()
            approx_G_t.append(loss.detach().cpu().item())

    approx_G = np.mean(approx_G_t)

    normD = netD.linear.weight.norm().cpu().item()
    normG= netG.map[1].weight.norm().cpu().item()
    print('[%d], D loss:%.4f, G loss:%.4f, G approx:%.4f, Ave Update:%.4f, G norm:%.4f, D norm:%.4f'  
         % (epoch, loss_D[-1], loss_G[-1], approx_G, grad_G[-1], normG , normD))

        
    # if epoch % 1 == 0:
    #     fig = plt.figure()
    #     plt.plot(loss_G, label='approximator loss')
    #     plt.xlabel('Epoch, update for each approximator G')
    #     plt.legend()
    #     fig.savefig('./loss_monitor/Approximator'+ str(epoch).zfill(4) + '.png')
    #     plt.close()

    # evaluation
    # show image
    if epoch % 20 == 0:
        
        netG.eval()
        show_z_b.normal_()
        ## show eval images
        fake_img = netG(show_z_b) #[N,1,28,28] torch.(cuda).tensor
        vutils.save_image((.5*fake_img+.5).detach().cpu() , './exp/'+ opt.exp + '/results/random/'+str(epoch).zfill(4)+'.png')
        ## show eval fixed images
        fake_img = netG(fix_show_z)
        vutils.save_image((.5*fake_img+.5).detach().cpu(), './exp/'+ opt.exp + '/results/fix/'+str(epoch).zfill(4)+'.png')

    # inception, diversity, FID scores.
    if epoch % 25 == 0:
        is_score.append(inception_score(netIncept, netG, device, opt.z_dim))
        with open('./exp/' + opt.exp + '/loss.txt', 'a') as append_File:
            print('[%d], Inception Score:%.4f, D loss:%.4f, G loss:%.4f, G approx:%.4f ,Ave Update:%.4f, G norm:%.4f, D norm:%.4f'  
                 % (epoch, is_score[-1], loss_D[-1], loss_G[-1], approx_G, grad_G[-1], normG , normD), file=append_File)
        print('IS score: %.4f' % is_score[-1])
        best_is = max(is_score[-1], best_is)

        fig = plt.figure()
        plt.plot(25 * (np.arange(epoch//25 + 1)), is_score, label='IS')
        plt.xlabel('Epoch, update for each approximator G')
        plt.legend()
        fig.savefig('./exp/'+ opt.exp +'/InceptionScore.png')
        plt.close()

        if best_is == is_score[-1]:
            print('Saving, with best is_score: %.4f' % is_score[-1])
            state = {
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'is_score': is_score,
                'loss_G': loss_G,
                'epoch': epoch,
                'best_is': best_is,
                'loss_D': loss_D,
                'grad_G': grad_G
                }
            torch.save(state, './checkpoint/cifar_exp_' + opt.exp +'.t7')
    ## print loss ?