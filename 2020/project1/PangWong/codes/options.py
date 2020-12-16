import sys
import argparse
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.parser = None
        self.isTrain = None

    def initialize(self, parser):
        parser.add_argument('--dataset', type=str, default='mnist')
        parser.add_argument('--dataroot', type=str, default='./datasets/')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='resnet18', help='which model to use')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize,
                            help='Maximum number of samples allowed per dataset.')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--load_size', type=int, default=224, help='Scale images to this size.')
        parser.add_argument('--num_workers', default=0, type=int, help='# threads for loading data')

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        opt = parser.parse_args()
        opt.isTrain = self.isTrain
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = [int(i) for i in str_ids if int(i) >= 0]
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        if not self.isTrain:
            opt.classifiers = opt.classifiers.split(',')

        self.parser = parser
        return opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=10, help='# of iterations.')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints')

        self.isTrain = True
        return parser


class EvalOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--which_epoch', type=str, help='Epoch # to load. Leave empty for pretrained weights')
        parser.add_argument('--classifiers', type=str, default='')
        # parser.add_argument('--num_splits', type=int, default=0)
        parser.add_argument('--use_cuml', action='store_true', help='Speed up classfiers with GPU using CuML.')
        parser.add_argument('--use_thundersvm', action='store_true', help='Speed up SVM with GPU using ThunderSVM.')
        # parser.add_argument('--tsne_pca_dim', type=int, default=50)
        # parser.add_argument('--tsne_perplexity', type=int, default=30)
        # parser.add_argument('--tsne_iter', type=int, default=1000)

        self.isTrain = False
        return parser
