#Some code are based on: https://github.com/schesho/Scattering-Conv-Network
import os
import mnist
import numpy as np
from scipy import signal, misc, ndimage
from itertools import combinations
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#This function is based on: https://github.com/schesho/Scattering-Conv-Network
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

#This function is based on: https://github.com/schesho/Scattering-Conv-Network
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

#This function is based on: https://github.com/schesho/Scattering-Conv-Network
def morl_conv(image, subsample = 1, J = 0, theta = 0, sigma = 0.85, xi = 3 * np.pi / 4):
    """image of shape (n_lines, n_columns) (levels of grey not RGB), returns convolution with Morlet2D_grid"""
    morlet_filter = Morlet2D_grid(image.shape[0], image.shape[1], J, theta, sigma, xi)
    return (signal.fftconvolve(image, morlet_filter, mode = "same")[ :: subsample,  ::subsample])

#This function is based on: https://github.com/schesho/Scattering-Conv-Network
def scatter_coeffs_fast(image, m, K, J = 0):
    """J = number of dilatations, m = length of the scattering conv network, K = number of rotations
        outputs a long array of concatenated scatter coefficients"""
    
    if J == 0 :
        #default value of J is log(image.width)
        J = int(math.log(image.shape[0], 2))
        
    #layer number 0: first scattering coeff
    scatter_coeffs = 1/2**J*ndimage.filters.gaussian_filter(image, 1/2**J)[ :: 2**J, :: 2**J].reshape(-1)
    
    #we index_pathes to normalize scattering coefficient afterwards
    path_index = 0
    #coeff indexes indicates which are the indexes corresponding to path n° path_index
    coeff_indexes = {path_index : (0, scatter_coeffs.shape[0])}
    #normalization_coeff[path_index ] = norm(S[path]x)
    normalization_coeff = {path_index : np.linalg.norm(scatter_coeffs)}
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
            #for path of length m calculate S[p]x, save path indexes in the scatter_coeffs array, and the normalizaiton_coeff
            path_index += 1
            new_coeff = (1/2**J*ndimage.filters.gaussian_filter(U_p[i][path], 1/2**J)[ :: 2**J, :: 2**J]).reshape(-1)
            scatter_coeffs = np.concatenate((scatter_coeffs, new_coeff))
            coeff_indexes[path_index] = (scatter_coeffs.shape[0] - new_coeff.shape[0], scatter_coeffs.shape[0])
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
        scatter_coeffs = np.concatenate((scatter_coeffs, new_coeff))
        coeff_indexes[path_index] = (scatter_coeffs.shape[0] - new_coeff.shape[0], scatter_coeffs.shape[0])
        normalization_coeff[path_index] = np.linalg.norm(new_coeff)
    return (scatter_coeffs, coeff_indexes, normalization_coeff)

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

