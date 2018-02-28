import torch
from scatwave.scattering import Scattering
import numpy as np
from contextlib import contextmanager
import sys, os
import warnings
warnings.filterwarnings("ignore")

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class scatNet(object):
    def __init__(self, size=(224, 224), layer=4):
        """initialize a scatNet instance (slow)"""
        self.scat = Scattering(M=size[0], N=size[1], J=layer).cuda()
    
    def inference(self, in_im):
        """inference on image. Input: numpy array of (batch_size, height, width, channels)"""
        in_im = np.transpose(in_im, (0, 3, 1, 2))
        in_im = torch.from_numpy(in_im).float().cuda()
        with suppress_stdout():
            return self.scat(in_im).cpu().numpy()




if __name__ == "__main__":
    import numpy as np
    scat = scatNet(size=(224,224), layer=4)
    x = np.random.randn(8,224,224,3)
    ret = scat.inference(x)
    assert(ret.shape == (8, 3, 417, 14, 14))
