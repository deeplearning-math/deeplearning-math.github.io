import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import mnist_reader

class Cluttered_MNIST(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx].reshape(1,40,40))
        label = torch.tensor(self.labels[idx][0], dtype=torch.int).type(torch.LongTensor)
        return [image, label]

class Fashion_MNIST(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).reshape(1,28,28).type(torch.FloatTensor)
        label = torch.tensor(self.labels[idx], dtype=torch.int).type(torch.LongTensor)
        return [image, label]

def get_fashion_train_valid_loader(data_dir,
                                     batch_size,
                                     random_seed,
                                     valid_size=0.1,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=False):
    
    X_dataset, y_dataset = mnist_reader.load_mnist(data_dir, kind='train')
    dataset = Fashion_MNIST(X_dataset, y_dataset)
    
    num_train = len(X_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    
    return (train_loader, valid_loader)

def get_fashion_test_loader(data_dir,
                             batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=False):
    
    X_test, y_test = mnist_reader.load_mnist(data_dir, kind='t10k')
    test_ds = Cluttered_MNIST(X_test, y_test)
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
    )

    return test_loader

    
def get_cluttered_train_valid_loader(data_dir,
                                     batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=False):
    
#     data_dir = '../mnist_sequence1_sample_5distortions5x5.npz'
    mnist_cluttered = np.load(data_dir)
    X_train = mnist_cluttered['X_train']
    y_train = mnist_cluttered['y_train']
    X_valid = mnist_cluttered['X_valid']
    y_valid = mnist_cluttered['y_valid']
    
    train_ds = Cluttered_MNIST(X_train, y_train)
    valid_ds = Cluttered_MNIST(X_valid, y_valid)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
    )
    
    return (train_loader, valid_loader)

def get_cluttered_test_loader(data_dir,
                             batch_size, 
                             shuffle=True,
                             num_workers=4,
                             pin_memory=False):
    
#     data_dir = '../mnist_sequence1_sample_5distortions5x5.npz'
    mnist_cluttered = np.load(data_dir)
    X_test = mnist_cluttered['X_test']
    y_test = mnist_cluttered['y_test']
    test_ds = Cluttered_MNIST(X_test, y_test)
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
    )

    return test_loader

def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=trans
    )

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
