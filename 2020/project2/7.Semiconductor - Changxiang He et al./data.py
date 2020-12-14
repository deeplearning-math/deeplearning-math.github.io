import os
import sys
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# set random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# transform PIL image to Tensor
transform = transforms.Compose([transforms.ToTensor()])

# class for building the dataset
class ImageReader(Dataset):
    def __init__(self, data, labels, c_mean=0.2154, c_std=0.1301):
        self.data = data
        self.labels = labels

        normalize = transforms.Normalize(mean = [c_mean,c_mean,c_mean],std = [c_std,c_std,c_std])
        self.all_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                normalize])

    def __getitem__(self, idx):
        """Returns one data pair (source and target)."""
        x = self.all_transforms(self.data[idx])
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)

# get data loader
def getDataLoader(dataset, bsz, test=False):
    if test:
        shuffle = False
    else:
        shuffle = True
    # prepare dataloader
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=bsz,
                                         shuffle=shuffle)
    return loader


def compute_normalize(matrix_train_x):
    means = []
    stds = []
    for i in range(len(matrix_train_x)):      
        tensor = transform(matrix_train_x[i]) 
        means.append(torch.mean(tensor))
        stds.append(torch.std(tensor))    
    c_mean = torch.mean(torch.tensor(means)) #算mean
    c_std = torch.mean(torch.tensor(stds)) #算std
    return c_mean, c_std

#get the image name list
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.bmp')]

#prepare the dataset
def load_data(path, train_ratio=0.8, split="train"): 
    print('return train/validation dataset')
    subdir = os.listdir(path) #['defect', 'good_all']
    matrix_train_x=[]
    matrix_train_y=[]

    for j in range(len(subdir)):
        filenames = get_imlist(path+'/'+subdir[j])
        if split == "train":
            train_number = int(len(filenames)*train_ratio)
            split_filenames = filenames[:train_number]
        elif split == "valid":
            train_number = int(len(filenames)*(1-train_ratio))
            split_filenames = filenames[-train_number:]
        else:
            split_filenames = filenames

        for name in tqdm(split_filenames, desc=f'Loading data', total=len(split_filenames), ncols=100):
            img = Image.open(name)  #打开图像
            img = img.resize((224,224))
            img = img.convert('RGB')
            matrix_train_x.append(img)
            if split == "test":
                matrix_train_y.append(0)
            else:
                matrix_train_y.append(0 if "defect" in subdir[j] else 1)
    
    return matrix_train_x, matrix_train_y


def get_loaders(train_path, test_path, batch_size, test_batch_size):
    train_data, train_labels = load_data(train_path, train_ratio=0.1, split="train")
    assert len(train_data) == len(train_labels)
    train_dataset = ImageReader(train_data, train_labels)
    train_loader = getDataLoader(train_dataset, batch_size, test=False)

    valid_data, valid_labels = load_data(train_path, train_ratio=0.8, split="valid")
    valid_dataset = ImageReader(valid_data, valid_labels)
    valid_loader = getDataLoader(valid_dataset, test_batch_size, test=True)

    test_data, test_labels = load_data(test_path, train_ratio=1, split="test")
    test_dataset = ImageReader(test_data, test_labels)
    test_loader = getDataLoader(test_dataset, test_batch_size, test=True)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # transform = transforms.Compose([            #[1]
    #             transforms.ToTensor(),                     #[4]
    #             transforms.Normalize(                      #[5]
    #             mean=[0.485, 0.456, 0.406],                #[6]
    #             std=[0.229, 0.224, 0.225]                  #[7]
    #             )])

    # # data
    # # labels
    # train_dataset = ImageReader(data, labels, transform)
    # train_loader = getDataLoader(train_dataset, 10, test=False)


    train_path = r"semi-conductor-image-classification-second-stage/train/train_contest"
    test_path = r"semi-conductor-image-classification-second-stage/test/test_contest"

    # (matrix_train_x,matrix_train_y) = load_data(path, train_ratio=0.8, split="train")
    # print(transform(matrix_train_x[0]))

    # mean, std = compute_normalize(matrix_train_x)
    # print("mean",mean, "std", std)

    # normalize = transforms.Normalize(mean = [mean,mean,mean],std = [std,std,std])
    # all_transforms = transforms.Compose([
    #                             #transforms.RandomHorizontalFlip(), # 对PIL.Image图片进行操作
    #                             transforms.ToTensor(),
    #                             normalize]) 
    # test_x = all_transforms(matrix_train_x[0]) #normalization
    # print(test_x)

    train_loader, valid_loader, test_loader = get_loaders(train_path, test_path, 2, 4)
    for item in train_loader:
        x, y = item
        print("x", x)
        print("y", y)
