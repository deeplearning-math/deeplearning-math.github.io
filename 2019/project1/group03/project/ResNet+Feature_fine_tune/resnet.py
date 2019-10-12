from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
import os
import numpy as np

class MNISTResNet18(ResNet):
    def __init__(self):
        super(MNISTResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(super(MNISTResNet18, self).forward(x), dim=-1)

def get_data_loaders(train_batch_size, val_batch_size):
    mnist = MNIST(download=True, train=True, root=".").train_data.float()
    
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def main():
    epochs = 5

    # ------- define model, loss and optimizer --------
    model = MNISTResNet18().cuda()

    losses = []
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())


    # ------- define dataloader ----------
    train_loader, val_loader = get_data_loaders(256, 256)

    batches = len(train_loader)
    val_batches = len(val_loader)
    
    # ------- start to train ----------
    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()
        
        for i, data in progress:
            X, y = data[0].cuda(), data[1].cuda()
            
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)

            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
            
        torch.cuda.empty_cache()
        
        # ------- Evaluate our model each epoch --------
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].cuda(), data[1].cuda()
                outputs = model(X)
                val_losses += loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]
                
                for acc, metric in zip((precision, recall, f1, accuracy), 
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )            
            
        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss/batches)
    print(losses)

if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
    os.system('rm tmp')
    main()

