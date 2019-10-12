import os
import mnist
import numpy as np
from scipy import signal, misc, ndimage
from itertools import combinations
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV




train_size = 1200
test_size = 1200

## Import train and test set
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images =  mnist.test_images()
test_labels = mnist.test_labels()


## parameters selection using grid search
param_grid = {'C': [1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

train_features = np.reshape(train_images[:train_size, :, :], [train_size, -1])
train_label = train_labels[:train_size]

clf1 = OneVsRestClassifier(GridSearchCV(svm.SVC(kernel='rbf'), param_grid), n_jobs=20)
clf_ = clf1.fit(train_features, train_label)



test_features = np.reshape(test_images[:test_size, :, :], [test_size, -1])
test_label = test_labels[:test_size]
y_pred = clf_.predict(test_features)
print('Accuracy score :', metrics.accuracy_score(y_pred, test_label))
print('Confusion matrix :', metrics.confusion_matrix(y_pred, test_label))

