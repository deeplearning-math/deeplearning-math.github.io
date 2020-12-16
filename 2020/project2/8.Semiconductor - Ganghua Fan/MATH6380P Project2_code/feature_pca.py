from kymatio.keras import Scattering2D

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tensorflow.keras import utils
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # 3D figure


import os
from cv2 import cv2
from PIL import Image
import pandas as pd


def get_files(file_dir):
    if not file_dir[-1] in '/\\':
        file_dir = file_dir + '\\'
    image_list = []
    imgs = []

    image_list = [file_dir + file_name for file_name in os.listdir(file_dir)]
  
    #img = [cv2.imread(i).resize(224,224) for i in image_list]
    for file_name in image_list:
        extension = file_name.split(".")[-1]
        if not extension in ['jpg', 'jpeg','bmp']:
            continue
        img = cv2.imread(file_name, 0)
        img2 = cv2.resize(img, (28,28))
        imgs.append(img2)
    return np.array(imgs)
    

def ScatFeature(x_train):
    scat_x_train = Scattering2D(J=2, L=8)(x_train)
    final_scat_x_train = Flatten()(scat_x_train)
    print(final_scat_x_train.shape)

    stand_x_train = StandardScaler().fit_transform(final_scat_x_train) # normalizing the features
    return stand_x_train


if __name__ == '__main__':
    # for good semiconductors
    train_dir = '/semiconductor/val/good_all'
    data = get_files(train_dir)
    stand_x_train = ScatFeature(data)
    

    pca = PCA(n_components=2)
    pca_o = pca.fit_transform(stand_x_train)
    pca_o.shape


    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(pca_o[:,0], pca_o[:,1], pca_o[:,2], marker='o')

    plt.scatter(pca_o[:,0], pca_o[:,1], marker='o')  # scattering
    # plt.legend('good_all')  
    #plt.show()

    # for defect semiconductors
    train_dir1 = '/semiconductor/val/defect'
    data1 = get_files(train_dir1)
    stand_x_train1 = ScatFeature(data1)
    

    pca = PCA(n_components=2)
    pca_o1 = pca.fit_transform(stand_x_train1)
    #pca_o1.shape
    # ax.scatter(pca_o1[:,0], pca_o1[:,1], pca_o1[:,2], marker='x')
    plt.scatter(pca_o1[:,0], pca_o1[:,1], marker='x')  # scattering 
    
    plt.legend(["good_all","defect"]) 
    plt.show()