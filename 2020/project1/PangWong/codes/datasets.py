import torch
import torch.utils.data
from torchvision import datasets, transforms as T


def create_dataloader(opt):
    train_dataset = ImageNetDataset(opt, train=True)
    test_dataset = ImageNetDataset(opt, train=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=opt.isTrain,
        num_workers=opt.num_workers,
        drop_last=opt.isTrain
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )

    return train_dataloader, test_dataloader


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, opt, train):
        # ImageNet statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform = T.Compose([
            T.Resize((opt.load_size, opt.load_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)])

        self.data = None
        if opt.dataset == 'mnist':
            self.data = datasets.MNIST(root=opt.dataroot, train=train, download=True, transform=self.transform)
        elif opt.dataset == 'fashion':
            self.data = datasets.FashionMNIST(root=opt.dataroot, train=train, download=True, transform=self.transform)
        else:
            raise NotImplementedError

        if opt.max_dataset_size < len(self.data):
            subset_indices = list(range(opt.max_dataset_size))
            self.data = torch.utils.data.Subset(self.data, subset_indices)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
