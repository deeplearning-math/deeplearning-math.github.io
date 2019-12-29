import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import numpy as np


def img_truncate(img):
	img_t = torch.clamp(img, -1, 1)
	return img_t


class PoolSet(Dataset):
    
    def __init__(self, p_z, p_img):
        ## input: torch.tensor (NOT CUDA TENSOR)
        self.len = len(p_z)
        self.z_data = p_z # [N, opt.z_dim]
        self.img_data = p_img # [N, 3, 32, 32]
    
    def __getitem__(self, index):
        return self.z_data[index], self.img_data[index]
    
    def __len__(self):
        return self.len

class MyShowSet(Dataset):

    def __init__(self, show_z):
        ## img is a cuda tensor
        self.len = len(show_z)
        self.z = show_z

    def __getitem__(self, index):
        return self.z[index]

    def __len__(self):
        return self.len

def inception_score(net, netG, device, z_dim, batch_size=250, eps=1e-6):
    
    net.to(device)
    netG.to(device)
    net.eval()
    netG.eval()


    pyx = np.zeros((batch_size*120, 10)) #[N, C]

    for i in range(120):

        eval_z_b = torch.randn(batch_size, z_dim).to(device)
        fake_img_b = netG(eval_z_b) #[N, 3, 32, 32], images generated from netG is in range[-1,1]
        #input of classifier net should range from [-1,1]
        pyx[i*batch_size: (i+1)*batch_size] = F.softmax(net(fake_img_b).detach(), dim=1).cpu().numpy() 

    py = np.mean(pyx, axis=0) #[C,]
    
    kl = np.sum(pyx * (np.log(pyx+eps) - np.log(py+eps)), axis=1)
    kl = kl.mean()
    
    return np.exp(kl)
