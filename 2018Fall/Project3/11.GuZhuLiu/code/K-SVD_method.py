# from time import time
import cv2
import argparse
# import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)
from operator import mul, sub
from skimage.util.shape import *
from skimage.util import pad
from functools import reduce
from math import floor, sqrt, log10
from scipy.sparse.linalg import svds
import timeit
import sys
import os
import criterion
import binary
import scipy.io as sio

# import scipy as sp
# import pdb
# from sklearn.feature_extraction.image import extract_patches_2d
# from sklearn.datasets import make_sparse_coded_signal
# from sklearn.decomposition import MiniBatchDictionaryLearning
# from sklearn.linear_model import OrthogonalMatchingPursuit
# from matplotlib import pyplot as plt


# tuned_parameters = [{'patch_size': [8, 12, 16, 20], 'window_stride': [3, 4, 5]}]
patch_size = 8

sigma = 20  # Noise standard dev.

window_shape = (patch_size, patch_size)

# window_shape = (patch_size, patch_size)  # Patches' shape
window_stride = 4  # Patches' step
dict_ratio = 0.1  # Ratio for the dictionary (training set).
num_dict = 1100
ksvd_iter = 10
max_sparsity = 1
max_resize_dim = 360
dict_train_blocks = 100000


# -------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------- PATCH CREATION -------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------#


def patch_matrix_windows(img, stride):
    # we return an array of patches(patch_size X num_patches)
    patches = view_as_windows(img, window_shape,
                              step=stride)  # shape = [patches in image row,patches in image col,rows in patch,cols in patch]
    # size of cond_patches = patch size X number of patches
    cond_patches = np.zeros((reduce(mul, patches.shape[2:4]), reduce(mul, patches.shape[0:2])))
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            cond_patches[:, j + patches.shape[1] * i] = np.concatenate(patches[i, j], axis=0)
    return cond_patches, patches.shape


def reconstruct_image(patch_final, noisy_image):
    img_out = np.zeros(noisy_image.shape)
    weight = np.zeros(noisy_image.shape)
    num_blocks = noisy_image.shape[0] - patch_size + 1
    for l in range(patch_final.shape[1]):
        i, j = divmod(l, num_blocks)
        temp_patch = patch_final[:, l].reshape(window_shape)
        # img_out[i, j] = temp_patch[1, 1]
        img_out[i:(i + patch_size), j:(j + patch_size)] = img_out[i:(i + patch_size), j:(j + patch_size)] + temp_patch
        weight[i:(i + patch_size), j:(j + patch_size)] = weight[i:(i + patch_size), j:(j + patch_size)] + np.ones(
            window_shape)

        # img_out = img_out/weight
    img_out = (noisy_image + 0.034 * sigma * img_out) / (1 + 0.034 * sigma * weight)

    print('max: ', np.max(img_out))

    return img_out


# -------------------------------------------------------------------------------------------------------------------#
# ------------------------------------------ APPROXIMATION PURSUIT METHOD : -----------------------------------------#
# ------------------------------------- MULTI-CHANNEL ORTHOGONAL MATCHING PURSUIT -----------------------------------#
# -------------------------------------------------------------------------------------------------------------------#

def omp(D, data, sparsity):
    max_error = sqrt(((sigma ** 1.15) ** 2) * data.shape[0])
    # max_coeff = D.shape[0]/2
    max_coeff = sparsity

    sparse_coeff = np.zeros((D.shape[1], data.shape[1]))
    tot_res = 0
    for i in range(data.shape[1]):
        count = floor((i + 1) / float(data.shape[1]) * 100)
        sys.stdout.write("\r- Sparse coding : Channel : %d%%" % count)
        sys.stdout.flush()

        x = data[:, i]
        res = x
        atoms_list = []
        res_norm = np.linalg.norm(res)
        temp_sparse = np.zeros(D.shape[1])

        while len(atoms_list) < max_coeff:  # and res_norm > max_error:
            proj = D.T.dot(res)
            i_0 = np.argmax(np.abs(proj))
            atoms_list.append(i_0)

            temp_sparse = np.linalg.pinv(D[:, atoms_list]).dot(x)
            res = x - D[:, atoms_list].dot(temp_sparse)
            res_norm = np.linalg.norm(res)

        tot_res += res_norm
        if len(atoms_list) > 0:
            sparse_coeff[atoms_list, i] = temp_sparse
    # print('\n', tot_res)
    # print('\r- Sparse coding complete.\n')

    return sparse_coeff


# -------------------------------------------------------------------------------------------------------------------#
# ------------------------------------------- DICTIONARY METHODS -------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------#


def dict_initiate(train_noisy_patches, dict_size):
    # dictionary intialization

    indexes = np.random.random_integers(0, train_noisy_patches.shape[1] - 1,
                                        dict_size)  # indexes of patches for dictionary elements
    dict_init = np.array(train_noisy_patches[:, indexes])  # each column is a new atom
    print(dict_init.shape)
    # dictionary normalization
    dict_init = dict_init - dict_init.mean()
    temp = np.diag(pow(np.sqrt(np.sum(np.multiply(dict_init, dict_init), axis=0)), -1))
    dict_init = dict_init.dot(temp)
    basis_sign = np.sign(dict_init[0, :])
    dict_init = np.multiply(dict_init, basis_sign)

    # print('Shape of dictionary : ', str(dict_init.shape) + '\n')
    # cv2.namedWindow('dict', cv2.WINDOW_NORMAL)
    # cv2.imshow('dict',dict_init.astype('double'))

    return dict_init


def dict_update(D, data, matrix_sparse, atom_id):
    indices = np.where(matrix_sparse[atom_id, :] != 0)[0]
    D_temp = D
    sparse_temp = matrix_sparse[:, indices]

    if len(indices) > 1:
        sparse_temp[atom_id, :] = 0

        matrix_e_k = data[:, indices] - D_temp.dot(sparse_temp)
        u, s, v = svds(np.atleast_2d(matrix_e_k), 1)
        D_temp[:, atom_id] = u[:, 0]
        matrix_sparse[atom_id, indices] = s.dot(v)

    return D_temp, matrix_sparse


# -------------------------------------------------------------------------------------------------------------------#
# ------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------#

def k_svd(train_noisy_patches, dict_size, sparsity):
    dict_init = dict_initiate(train_noisy_patches, dict_size)

    D = dict_init

    matrix_sparse = np.zeros((D.T.dot(train_noisy_patches)).shape)  # initializing spare matrix
    num_iter = ksvd_iter
    # print('\nK-SVD, with residual criterion.')
    # print('-------------------------------')

    for k in range(num_iter):
        # print("Stage ", str(k + 1), "/", str(num_iter), "...")

        matrix_sparse = omp(D, train_noisy_patches, sparsity)

        count = 1

        dict_elem_order = np.random.permutation(D.shape[1])

        for j in dict_elem_order:
            r = floor(count / float(D.shape[1]) * 100)
            # sys.stdout.write("\r- Dictionary updating : %d%%" % r)
            # sys.stdout.flush()

            D, matrix_sparse = dict_update(D, train_noisy_patches, matrix_sparse, j)
            count += 1
        # print('\r- Dictionary updating complete.\n')

    return D, matrix_sparse


# -------------------------------------------------------------------------------------------------------------------#
# ------------------------------------------------ DENOISING METHOD -------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------#


def denoising(noisy_image, dict_size, sparsity):
    # 1. Form noisy patches.
    padded_noisy_image = pad(noisy_image, pad_width=window_shape, mode='symmetric')

    # dictionary intialization

    poss_patches = (noisy_image.shape[0] - patch_size + 1) * (noisy_image.shape[1] - patch_size + 1)
    stride = floor(poss_patches / dict_train_blocks)
    print(stride)
    if stride < 1:
        stride = 1
    # print('img_patches: ', poss_patches, 'train_stride: ', stride)
    stride = 2
    train_noisy_patches, train_noisy_patches_shape = patch_matrix_windows(padded_noisy_image, stride)
    train_data_mean = train_noisy_patches.mean()
    train_noisy_patches = train_noisy_patches - train_data_mean  # X(size mxp) = D(size mxn) x matrix_sparse(size nxp)

    # 3. Compute K-SVD.
    start = timeit.default_timer()
    dict_final, sparse_init = k_svd(train_noisy_patches, dict_size, sparsity)
    # stop = timeit.default_timer()
    # print ("Calculation time : " , str(stop - start) , ' seconds.')

    noisy_patches, noisy_patches_shape = patch_matrix_windows(padded_noisy_image, stride=1)
    data_mean = noisy_patches.mean()
    noisy_patches = noisy_patches - data_mean

    # start = timeit.default_timer()
    sparse_final = omp(dict_final, noisy_patches, sparsity)

    # 4. Reconstruct the image.
    patches_approx = dict_final.dot(sparse_final) + data_mean

    padded_denoised_image = reconstruct_image(patches_approx, padded_noisy_image)
    # patches_approx = patches_approx.reshape(noisy_patches.shape[1], *(patch_size,patch_size))
    # padded_denoised_image = reconstruct_from_patches_2d(patches_approx, (padded_noisy_image.shape[0]//2, padded_noisy_image.shape[1]//2))

    stop = timeit.default_timer()
    # print("Calculation time : ", str(stop - start), ' seconds.')

    shrunk_0, shrunk_1 = tuple(map(sub, padded_denoised_image.shape, window_shape))
    denoised_image = np.abs(padded_denoised_image)[window_shape[0]:shrunk_0, window_shape[1]:shrunk_1]

    return denoised_image, stop - start


def MSE(matrixA, matrixB):
    return pow(np.linalg.norm(matrixA - matrixB), 2) / matrixA.shape[0] / matrixA.shape[1]


def denoise(image):
    denoised_image = denoising(image, dict_size=num_dict, sparsity=max_sparsity)
    return denoised_image


# f = open('psnrVSnum_dict.txt','w')
# f.write('dict_size' + '\tnoisy_psnr' + '\tfinal_psnr')
# f = open('psnrVSsparsity.txt','w')
# f.write('max_sparsity' + '\tnoisy_psnr' + '\tfinal_psnr')

# for max_sparsity in range(1,11,1):
# print('num_dict:', num_dict, 'max_sparsity:', max_sparsity)

path = 'D:\cryo_em_data\EMDB2660_10028\ww_process_20180915_cryoSPARC\prepare_most_evident_images_for_xianyin\\raw_particles'
path1 = os.listdir(path)
i = 0
for filename in path1:
    new_path = os.path.join(path, filename)
    image = cv2.imread(new_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_init_size = max(image.shape[0], image.shape[1])
    resize_ratio = max_resize_dim / max_init_size

    image = image * 1.0

    if resize_ratio < 1:
        image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
    noisy_image = image
    denoised_image, calc_time = denoising(noisy_image, dict_size=num_dict, sparsity=max_sparsity)
    cv2.imwrite('D:\cryo_em\code\data\ksvd_embd10028\\' + filename, denoised_image)
    i = i +1
    print(i)
    #new_image = binary.findGoldMask(denoised_image) * 255
    # print("ok")
    # print(new_image)
    # cv2.imwrite('D:\cryo_em\code\denoise_code\\new_image.png', new_image)
    #new_orig = binary.findGoldMask(original_image) * 255
    #cv2.imwrite('D:\cryo_em\code\denoise_code\\new_orig.png', new_orig)
    #print(criterion.MSE(new_image / 255, new_orig / 255))

##calculate PSNR

'''new_path = 'D:\cryo_em\code\data\ddtf_new\\img005.mat'
image = cv2.imread(new_path)
original_image = cv2.imread('D:\cryo_em\code\data\clean_images\\000041@center_str3_align_128pixel_50000.png_1.png')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
max_init_size = max(image.shape[0], image.shape[1])
resize_ratio = max_resize_dim / max_init_size

image = image * 1.0
if resize_ratio < 1:
    image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
noisy_image = image
# clf  = GridSearchCV(denoising(), tuned_parameters, cv=5,
# ='%s_weighted' % score)
for i in range(len(patch_size1)):
    for j in range(len(sigma1)):
        patch_size = patch_size1[i]
        sigma = sigma1[j]
        window_shape = (patch_size, patch_size)
        denoised_image, calc_time = denoising(noisy_image, dict_size=num_dict, sparsity=max_sparsity)
        print('patch_size:%d,sigma:%d' % (patch_size, sigma))
        # cv2.imwrite('D:\cryo_em\code\denoise_code\\denoise_image.png', denoised_image)
        # new_image = binary.findGoldMask(denoised_image) * 255
        # print("ok")
        # print(new_image)
        # cv2.imwrite('D:\cryo_em\code\denoise_code\\new_image.png', new_image)
        # new_orig = binary.findGoldMask(original_image) * 255
        # cv2.imwrite('D:\cryo_em\code\denoise_code\\new_orig.png', new_orig)
        print(criterion.MSE(denoised_image / 255, original_image / 255))'''
