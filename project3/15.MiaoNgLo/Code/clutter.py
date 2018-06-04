import numpy as np
import torch

def translate(batch):
    batch1 = batch.cpu().data.unsqueeze(1).numpy()
    n, c, w_i = batch1.shape[:3]
    w_o = 60
    data = np.zeros(shape=(n, c, w_o, w_o), dtype=np.float32)
    for k in range(n):
        i, j = np.random.randint(0, w_o - w_i, size=2)
        data[k, :, i:i + w_i, j:j + w_i] += batch1[k]
    return torch.from_numpy(data).type_as(batch).squeeze(1)


def clutter(batch):
    batch1 = batch.cpu().data.unsqueeze(1).numpy()
    n, c, w_i = batch1.shape[:3]
    w_o = 60
    data = np.zeros(shape=(n, c, w_o, w_o), dtype=np.float32)
    for k in range(n):
        i, j = np.random.randint(0, w_o - w_i, size=2)
        data[k, :, i:i + w_i, j:j + w_i] += batch1[k]
        for _ in range(4):
            clt = batch1[np.random.randint(0, batch1.shape[0] - 1)]
            c1, c2 = np.random.randint(0, w_i - 8, size=2)
            i1, i2 = np.random.randint(0, w_o - 8, size=2)
            data[k, :, i1:i1 + 8, i2:i2 + 8] += clt[:, c1:c1 + 8, c2:c2 + 8]
    data = np.clip(data, 0., 1.)
    return torch.from_numpy(data).type_as(batch).squeeze(1)