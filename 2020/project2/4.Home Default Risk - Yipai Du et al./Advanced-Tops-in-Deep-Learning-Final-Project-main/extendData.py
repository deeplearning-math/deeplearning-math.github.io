from collections import Counter

import imblearn
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X_train = np.load('data/train_X_ori.npy').astype('float32')
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    y_train = np.load('data/train_y_ori.npy').astype('float32').reshape(
        (-1, 1))

    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                          y_train,
                                                          test_size=0.2,
                                                          random_state=0)
    np.save("data/mean.npy", mean)
    np.save("data/std.npy", std)
    np.save("data/X_valid.npy", X_valid)
    np.save("data/y_valid.npy", y_valid)
    np.save("data/X_train_no_ext.npy", X_train)
    np.save("data/y_train_no_ext.npy", y_train)
