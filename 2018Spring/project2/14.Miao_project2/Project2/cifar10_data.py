"""
cifar-10 dataset, with support for random labels
"""
import numpy as np

import torch
import torchvision.datasets as datasets


class CIFAR10Random(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, shuffle_prob=0.0, random_prob=0.0, mosaic_prob=0.0, Gaussian_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10Random, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)
    if shuffle_prob > 0:
      self.shuffle_pixels(shuffle_prob)
    if random_prob > 0:
      self.random_pixels(random_prob)
    if Gaussian_prob > 0:
      self.Gaussian_pixels(Gaussian_prob)
    if mosaic_prob > 0:
      self.mosaic_pixels(mosaic_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.train_labels if self.train else self.test_labels)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    if self.train:
      self.train_labels = labels
    else:
      self.test_labels = labels

  def shuffle_row_col(self):
    data = self.train_data if self.train else self.test_data
    np.random.seed(12345)
    rand_permute_1 = np.random.permutation(np.identity(32)).astype(np.uint8)
    rand_permute_2 = np.random.permutation(np.identity(32)).astype(np.uint8)
    data = np.tensordot(data, rand_permute_1, axes=((1), (1)))
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 2, 3)
    data = np.tensordot(data, rand_permute_2, axes=((2), (0)))
    data = np.swapaxes(data, 2, 3)
    data = data.astype(np.uint8)

    if self.train:
      self.train_data = data
    else:
      self.test_data = data

  def random_row_col(self):
    data = self.train_data if self.train else self.test_data
    for i in range(data.shape[0]):
      np.random.seed(12345)
      rand_permute_1 = np.random.permutation(np.identity(32)).astype(np.uint8)
      rand_permute_2 = np.random.permutation(np.identity(32)).astype(np.uint8)
      data[i] = np.swapaxes(np.swapaxes(np.tensordot(data[i], rand_permute_1, axes=((0), (1))), 0, 2), 1, 2)
      data[i] = np.swapaxes(np.tensordot(data[i], rand_permute_2, axes=((1), (0))), 1, 2)
      data[i] = data[i].astype(np.uint8)

    if self.train:
      self.train_data = data
    else:
      self.test_data = data

  def shuffle_pixels(self, shuffle_prob):
    data = self.train_data if self.train else self.test_data
    np.random.seed(12345)
    mask = np.random.rand(32, 32) <= shuffle_prob
    num = np.size(np.zeros((32, 32))[mask])
    rand_permute = np.random.permutation(np.identity(num))
    for i in range(data.shape[0]):
      tmp = data[i, mask, :]
      tmp = np.dot(rand_permute, tmp)
      data[i, mask, :] = tmp
      data[i] = data[i].astype(np.uint8)

    if self.train:
      self.train_data = data
    else:
      self.test_data = data

  def random_pixels(self, random_prob):
    data = self.train_data if self.train else self.test_data
    for i in range(data.shape[0]):
      np.random.seed(12345)
      mask = np.random.rand(32, 32) <= random_prob
      tmp = data[i, mask, :]
      np.random.shuffle(tmp)
      data[i, mask, :] = tmp

    if self.train:
      self.train_data = data
    else:
      self.test_data = data

  def Gaussian_pixels(self, Gaussian_prob):
    data = self.train_data if self.train else self.test_data
    for i in range(data.shape[0]):
      np.random.seed(12345)
      mask = np.random.rand(32, 32) <= Gaussian_prob
      for j in range(data.shape[3]):
          mean = np.mean(data[i,:,:,j])
          std = np.std(data[i,:,:,j])
          data[i, mask, j] = np.random.normal(loc=mean, scale=std, size=(32, 32))[mask]

  def mosaic_pixels(self, mosaic_prob):
    data = self.train_data if self.train else self.test_data
    mean = np.zeros(3)
    std = np.zeros(3)
    for j in range(data.shape[3]):
      mean[j] = np.mean(data[:, :, :, j])
      std[j] = np.std(data[:, :, :, j])
    for i in range(data.shape[0]):
      np.random.seed(12345)
      if np.random.rand() <= mosaic_prob:
        for j in range(data.shape[3]):
          data[i, :, :, j] = np.random.normal(loc=mean, scale=std, size=(32, 32))

    if self.train:
      self.train_data = data
    else:
      self.test_data = data