import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import get_test_loader, get_train_valid_loader, get_cluttered_train_valid_loader, get_cluttered_test_loader, get_fashion_train_valid_loader, get_fashion_test_loader


def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    # instantiate data loaders
    clutter_mnist_path = '../mnist_sequence1_sample_5distortions5x5.npz'
    fashion_mnist_dir = '../data/fashion'
    if config.is_train:
#         data_loader = get_cluttered_train_valid_loader(clutter_mnist_path, config.batch_size,
#             config.valid_size, config.shuffle)
        data_loader = get_fashion_train_valid_loader(
            fashion_mnist_dir, config.batch_size,
            config.random_seed, config.valid_size,
            config.shuffle, config.show_sample, **kwargs)
#         data_loader = get_train_valid_loader(
#             config.data_dir, config.batch_size,
#             config.random_seed, config.valid_size,
#             config.shuffle, config.show_sample, **kwargs)
    else:
#         data_loader = get_cluttered_test_loader(clutter_mnist_path, config.batch_size,
#             config.valid_size, config.shuffle)
        data_loader = get_fashion_test_loader(
            fashion_mnist_dir, config.batch_size, **kwargs)
#         data_loader = get_test_loader(
#             config.data_dir, config.batch_size, **kwargs)
    
    
    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
