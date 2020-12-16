import os
import torch


def save_network(net, epoch, opt):
    ckpt_fn = '{}_{}_epoch{}.pth'.format(opt.dataset, opt.model, epoch)
    ckpt_path = os.path.join(opt.checkpoints_dir, ckpt_fn)
    torch.save(net.cpu().state_dict(), ckpt_path)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        net.cuda()


def load_network(net, opt):
    ckpt_fn = '{}_{}_epoch{}.pth'.format(opt.dataset, opt.model, opt.which_epoch)
    ckpt_path = os.path.join(opt.checkpoints_dir, ckpt_fn)
    weights = torch.load(ckpt_path)
    net.load_state_dict(weights, strict=False)
    return net
