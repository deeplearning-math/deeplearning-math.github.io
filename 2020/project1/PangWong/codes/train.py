from tqdm import tqdm
import torch
import time

from options import TrainOptions
from datasets import create_dataloader
from models import create_model
from utils.util import save_network


def train_one_step(model, optim, train_batch):
    optim.zero_grad()
    inputs_batch, labels_batch = train_batch
    logits, loss, acc = model({
        'inputs': inputs_batch,
        'labels': labels_batch
    }, mode='train')
    loss.backward()
    optim.step()

    metrics = {
        'loss': '{:.5f}'.format(loss.data.item()),
        'acc': '{:.3%}'.format(acc.data.item())
    }
    return metrics


def validate_one_step(model, val_batch):
    inputs_batch, labels_batch = val_batch
    losses_sum, accs_sum = model({
        'inputs': inputs_batch,
        'labels': labels_batch
    }, mode='val')

    return losses_sum, accs_sum


def train_data():
    opt = TrainOptions().parse()
    train_dataloader, test_dataloader = create_dataloader(opt)
    model = create_model(opt)
    optim = model.create_optimizer(opt)

    print('Start training.')
    width = len(str(opt.niter))
    for it in range(1, opt.niter+1):
        it_str = '{:0{width}d}'.format(it, width=width)
        tqdm_desc = 'Epoch #{}/{:d}'.format(it_str, opt.niter)

        iter_start_time = time.time()
        train_pbar = tqdm(train_dataloader, desc=tqdm_desc)
        for data_i in train_pbar:
            metrics = train_one_step(model, optim, data_i)
            train_pbar.set_postfix(metrics)

        val_losses = []
        val_accs = []
        for data_i in test_dataloader:
            losses_sum, accs_sum = validate_one_step(model, data_i)
            val_losses.append(losses_sum)
            val_accs.append(accs_sum)

        val_size = len(test_dataloader.dataset)
        ave_val_loss = torch.stack(val_losses).sum().data.item() / val_size
        ave_val_acc = torch.stack(val_accs).sum().data.item() / val_size
        print('test loss: {:.5f}, test accuracy: {:.3%}, time elapsed {:.3f}s'.format(
            ave_val_loss, ave_val_acc, time.time() - iter_start_time
        ))

        if it % opt.save_epoch_freq == 0:
            save_network(model, it_str, opt)


if __name__ == '__main__':
    train_data()
