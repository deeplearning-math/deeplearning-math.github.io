import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.feature_selection import RFECV

X_train = np.load("data/X_train_no_ext.npy")
y_train = np.load("data/y_train_no_ext.npy").squeeze()
X_valid = np.load("data/X_valid.npy")
y_valid = np.load("data/y_valid.npy").squeeze()

clf = TabNetClassifier()
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
np.save("data/importance.npy", clf.feature_importances_)
print((clf.feature_importances_ == 0.0).sum())
print((clf.feature_importances_ != 0.0).sum())

X_test = np.load("data/test_X.npy", allow_pickle=True)
mean = np.load("data/mean.npy", allow_pickle=True)
std = np.load("data/std.npy", allow_pickle=True)
X_test = (X_test - mean) / std
preds = clf.predict(X_test)
tabnet_out = pd.read_csv("data/test_id.csv")
tabnet_out['TARGET'] = preds
tabnet_out.to_csv("data/tabnet_submit.csv", index=False, header=True)
