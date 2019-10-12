#Some code are based on: https://github.com/schesho/Scattering-Conv-Network
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


def Morlet2D_grid(n_lines, n_columns, J = 0, theta = 0, sigma = 0.85, xi = 3 * np.pi / 4):
    """returns 2d Morlet real and imaginary filter of size (n_lines, n_columns), rotated by theta and dilated by 1/2^J"""
    X = np.arange(-n_columns / 2**(J+1), n_columns / 2**(J+1), 1/2**J) #to get n_columns length vector
    Y = np.arange(-n_lines / 2**(J+1), n_lines / 2**(J+1), 1/2**J)
    X, Y = np.meshgrid(X, Y)
    
    # Rotate
    X_theta = np.cos(theta) * X + np.sin(theta) * Y 
    Y_theta = np.cos(theta) * Y - np.sin(theta) * X
    
    Wave = np.exp(1j * xi * (X_theta + Y_theta))
    Z_r = np.real(Wave)
    Z_i = np.imag(Wave)
    Gaus = np.exp(-np.sqrt(X_theta**2 + Y_theta**2) / (2 * sigma**2))
    Z_r *= Gaus
    Z_i *= Gaus
    
    # Center Morlet wave
    Int_r = np.sum(Z_r)
    Int_i = np.sum(Z_i)
    Int_Gaus = np.sum(Gaus)

    beta_r = Int_r / Int_Gaus
    beta_i = Int_i/ Int_Gaus
    
    Z_r -= beta_r * Gaus
    Z_i -= beta_i * Gaus
    return(Z_r + 1j * Z_i)

def continue_path(path, J, K):
    """gets a scattering path of length m, outputs a list of all paths of length m+1 that you could obtain from this path"""
    cont_paths = []
    
    if type(path[0]) == int: #du au fait qu'un tuple ne peut pas avoir une longueur de 1
        j_m = path[0]
    else:
        j_m = path[-1][0]
    thetas = [np.pi * k / K for k in range(K)]
    for j in range(j_m + 1, J + 1):
        for theta in thetas:
            if type(path[0]) == int:
                path1 = [path]
            else:
                path1 = list(path)
            path1 += [(j, theta)]
            cont_paths.append(tuple(path1))
    return cont_paths


def morl_conv(image, subsample = 1, J = 0, theta = 0, sigma = 0.85, xi = 3 * np.pi / 4):
    """image of shape (n_lines, n_columns) (levels of grey not RGB), returns convolution with Morlet2D_grid"""
    morlet_filter = Morlet2D_grid(image.shape[0], image.shape[1], J, theta, sigma, xi)
    return (signal.fftconvolve(image, morlet_filter, mode = "same")[ :: subsample,  ::subsample])

def scatter_coeffs_fast(image, m, K, J = 0):
    """J = number of dilatations, m = length of the scattering conv network, K = number of rotations
        outputs a long array of concatenated scatter coefficients"""
    
    if J == 0 :
        #default value of J is log(image.width)
        J = int(math.log(image.shape[0], 2))
        
    #layer number 0: first scattering coeff
    scat_coeffs = 1/2**J*ndimage.filters.gaussian_filter(image, 1/2**J)[ :: 2**J, :: 2**J].reshape(-1)
    
    #we index_pathes to normalize scattering coefficient afterwards
    path_index = 0
    #coeff indexes indicates which are the indexes corresponding to path n° path_index
    coeff_indexes = {path_index : (0, scat_coeffs.shape[0])}
    #normalization_coeff[path_index ] = norm(S[path]x)
    normalization_coeff = {path_index : np.linalg.norm(scat_coeffs)}
    paths = []
    U_p = {1 : {}}
    #first layers U[p] calculated
    for j in range(1, J +1):
        for theta in [np.pi * k / K for k in range(K)]:
            path = (j, theta)
            paths.append((path))
            U_p[1][path] = np.abs(morl_conv(image, subsample = path[0], J = path[0], theta = path[1]))
    #fast scattering transform:
    for i in range(1, m):
        U_p[i+1] = {}
        paths2 = []
        for path in paths:
            #for path of length m calculate S[p]x, save path indexes in the scat_coeffs array, and the normalizaiton_coeff
            path_index += 1
            new_coeff = (1/2**J*ndimage.filters.gaussian_filter(U_p[i][path], 1/2**J)[ :: 2**J, :: 2**J]).reshape(-1)
            scat_coeffs = np.concatenate((scat_coeffs, new_coeff))
            coeff_indexes[path_index] = (scat_coeffs.shape[0] - new_coeff.shape[0], scat_coeffs.shape[0])
            normalization_coeff[path_index] = np.linalg.norm(new_coeff)
            for next_path in continue_path(path, J, K):
                #calculate all paths of length m+1 and U[p] for them
                U_p[i+1][next_path] =  np.abs(morl_conv(image, subsample = next_path[-1][0], J = next_path[-1][0], theta = next_path[-1][1]))
                paths2.append(next_path)
        paths = paths2
    for path in paths:
        #for remaining paths of length m, do the first step of the previous loop
        path_index += 1
        new_coeff = 1/2**J*ndimage.filters.gaussian_filter(U_p[m][path], 1/2**J).reshape(-1)#
        scat_coeffs = np.concatenate((scat_coeffs, new_coeff))
        coeff_indexes[path_index] = (scat_coeffs.shape[0] - new_coeff.shape[0], scat_coeffs.shape[0])
        normalization_coeff[path_index] = np.linalg.norm(new_coeff)
    return (scat_coeffs, coeff_indexes, normalization_coeff)

def get_Scatter_features(mnist_images, idxs=[0]):
    train_set = [np.array(mnist_images[idx]) for idx in idxs]
    sample_size = len(train_set)

    X = []
    d = {}
    for i in range(sample_size):
        if i % 100 == 0:
            print(i)
        scat_coeff, coeff_index, norm_coeff = scatter_coeffs_fast(train_set[i], m=2, K=6, J=3)
        X.append(scat_coeff)
        for i in norm_coeff:
            if norm_coeff[i] > d.get(i,0):
                d[i] = norm_coeff[i]

    X = np.array(X)      
    return X

train_size = 1200
test_size = 1200

## Import train and test set
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images =  mnist.test_images()
test_labels = mnist.test_labels()





## Using a small dataset for parameters selection using grid search
param_grid = {'C': [1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

small_dataset_size = 100
idxs_small = range(0, small_dataset_size)
train_features = get_Scatter_features(train_images, idxs_small)# Nxvector
train_label = train_labels[:small_dataset_size]
clf_small =  GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
clf = clf_small.fit(train_features, train_label)
print("Best estimator found by grid search : ", clf.best_estimator_)
### C = 100 gamma 1e-9


## Training
idxs = range(0, train_size)
train_features = get_Scatter_features(train_images, idxs)# Nxvector
train_label = train_labels[:train_size]

## Parallel train SVM 
clf0 =  OneVsRestClassifier(svm.SVC(kernel='rbf', C=100, gamma=1e-9), n_jobs=20)
clf = clf0.fit(train_features, train_label)

print("Train finished")

## Testing
idxs = range(0, test_size)
test_features = get_Scatter_features(test_images, idxs)# Nxvector
test_label = test_labels[:test_size]
y_pred = clf.predict(test_features)
print('Accuracy score :', metrics.accuracy_score(y_pred, test_label))
print('Confusion matrix :', metrics.confusion_matrix(y_pred, test_label))
