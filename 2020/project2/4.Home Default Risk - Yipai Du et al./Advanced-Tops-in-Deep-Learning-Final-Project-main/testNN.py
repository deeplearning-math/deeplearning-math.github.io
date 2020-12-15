import numpy as np
import pandas as pd
import torch
from trainNN import NN
model = torch.load("data/NN_model.pt").cuda()
X_test = np.load("data/test_X.npy", allow_pickle=True).astype('float32')
mean = np.load("data/mean.npy", allow_pickle=True)
std = np.load("data/std.npy", allow_pickle=True)
X_test = torch.from_numpy((X_test - mean) / std)
# importance = np.load("data/importance.npy", allow_pickle=True)
# ind = np.where(importance < 1e-10)
# X_test = np.delete(X_test, ind, axis=1)
model.eval()
with torch.no_grad():
    preds = model(X_test.cuda())
NN_out = pd.read_csv("data/test_id.csv")
NN_out['TARGET'] = preds.detach().cpu().numpy()
NN_out.to_csv("data/NN_submit.csv", index=False, header=True)