import numpy as np
from skimage.exposure import rescale_intensity
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt


def MSE(matrixA, matrixB):
    matrixB = rescale_intensity(matrixB, out_range=(0, 1))
    matrixA = rescale_intensity(matrixA, out_range=(0, 1))
    return pow(np.linalg.norm(matrixA - matrixB), 2) / matrixA.shape[0] / matrixA.shape[1]


def conduction(n, k):
    return 1 - pow(k, 2) / (pow(k, 2) + pow(n, 2))


def GCMSE(matrixA, matrixB, k):
    a = 0
    length = matrixA.shape[0]
    width = matrixB.shape[0]
    G = np.zeros([length, width])
    for i in range(length - 2):
        for j in range(width - 2):
            N = abs(matrixB[i + 1][j + 1] - matrixB[i + 1][j])
            S = abs(matrixB[i + 1][j + 1] - matrixB[i + 1][j + 2])
            W = abs(matrixB[i + 1][j + 1] - matrixB[i][j + 1])
            E = abs(matrixB[i + 1][j + 1] - matrixB[i + 2][j + 1])
            G[i][j] = (conduction(N, k) + conduction(S, k) + conduction(W, k) + conduction(E, k)) / 4
    for i in range(length):
        for j in range(width):
            a = a + pow((matrixA[i][j] - matrixB[i][j]) * G[i][j], 2)
    return a / (k + np.sum(G))


'''data = sio.loadmat('D:\cryo_em\code\data\ddtf_new\img08.mat')
data = data['img']
image = data[:, :, 0]
plt.imshow(image, cmap='gray')
plt.show()
image = rescale_intensity(1.0 * image, out_range=(0, 1))
image = binary.findGoldMask(image)
plt.imshow(image, cmap='gray')
plt.show()
path = 'D:\cryo_em\code\data\clean_images\\000041@center_str3_align_128pixel_50000.png_1.png'
image_temp = cv2.imread(path)
image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
image_temp = rescale_intensity(1.0 * image_temp, out_range=(0, 1))  # rescale the images
image_temp = binary.findGoldMask(image_temp)
plt.imshow(image_temp, cmap='gray')
plt.show()
# new_image = sio.loadmat('D:\matlab\\bin\\recon100_color_005')
new_image = sio.loadmat('D:\cryo_em\code\data\CWF\\recon100_08')
new_image = new_image['recon']
new_image = new_image[:, :, 0]
new_image = rescale_intensity(1.0 * new_image, out_range=(0, 1))
new_image = binary.findGoldMask(new_image)
plt.imshow(new_image, cmap='gray')
plt.show()
print(MSE(image, image_temp))
print(MSE(new_image, image_temp))'''
