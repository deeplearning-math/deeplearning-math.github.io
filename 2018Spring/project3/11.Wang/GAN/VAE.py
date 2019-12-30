import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class decoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset='mnist'):
        super(decoder, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 62
            self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

class encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist'):
        super(encoder, self).__init__()
        self.z_dim = 62
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)

        return x
class VAE_T(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE_T, self).__init__()
        self.z_dim = 64
        self.encoder = encoder
        self.decoder = decoder

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = h_enc[:, :62]
        log_sigma = h_enc[:, 62:]
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).type(torch.FloatTensor).cuda()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class VAE(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type

        # networks init
        self.En = encoder(self.dataset)
        self.De = decoder(self.dataset)
        self.VAE = VAE_T(self.En, self.De)
        self.VAE_optimizer = optim.Adam(self.VAE.parameters(),lr=args.lrG, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.VAE.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.De)
        utils.print_network(self.En)
        print('-----------------------------------------------')

        # load dataset
        if self.dataset == 'mnist':
            self.data_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=False,
                                                                          transform=transforms.Compose(
                                                                              [transforms.ToTensor()])),
                                                           batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'fashion-mnist':
            self.data_loader = DataLoader(
                datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=self.batch_size, shuffle=True)

        self.z_dim = 62
        self.z_n = utils.gaussian(self.batch_size, self.z_dim)

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.from_numpy(self.z_n).type(torch.FloatTensor).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.from_numpy(self.z_n).type(torch.FloatTensor), volatile=True)

    def train(self):
        self.train_hist = {}
        self.train_hist['VAE_loss'] = []
        self.train_hist['KL_loss'] = []
        self.train_hist['LL_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.VAE.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.En.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.from_numpy(self.z_n).type(torch.FloatTensor)

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                # update VAE network
                dec = self.VAE(x_)
                self.VAE_optimizer.zero_grad()
                KL_loss = latent_loss(self.VAE.z_mean, self.VAE.z_sigma)
                LL_loss = self.BCE_loss(dec, x_)
                VAE_loss = LL_loss + KL_loss
                self.train_hist['VAE_loss'].append(VAE_loss.data[0])
                self.train_hist['KL_loss'].append(KL_loss.data[0])
                self.train_hist['LL_loss'].append(LL_loss.data[0])
                VAE_loss.backward()
                self.VAE_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f"%
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size,
                           time.time() - start_time,VAE_loss.data[0]))
                if np.mod((iter + 1), 300) == 0:
                    samples = self.De(self.sample_z_)
                    if self.gpu_mode:
                        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
                    else:
                        samples = samples.data.numpy().transpose(0, 2, 3, 1)
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    utils.save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                utils.check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                                '_train_{:02d}_{:04d}.png'.format(epoch, (iter + 1)))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.model_dir + '/' + self.model_name, self.epoch)
        utils.loss_VAE_plot(self.train_hist, os.path.join(self.save_dir, self.model_dir), self.model_name)

    def visualize_results(self, epoch, fix=False):
        self.De.eval()

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.De(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.from_numpy(self.z_n).type(torch.FloatTensor).cuda(), volatile=True)
            else:
                sample_z_ = Variable(torch.from_numpy(self.z_n).type(torch.FloatTensor), volatile=True)

            samples = self.De(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          utils.check_folder(self.result_dir + '/' + self.model_dir) + '/' +
                          self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset,
            self.batch_size, self.z_dim)

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.VAE.state_dict(), os.path.join(save_dir, self.model_name + '_VAE.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.VAE.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_VAE.pkl')))