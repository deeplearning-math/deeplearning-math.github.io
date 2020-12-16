from kymatio.keras import Scattering2D

import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense

from sklearn.preprocessing import StandardScaler
import numpy as np


from tensorflow.keras import utils
import matplotlib.pyplot as plt

import os
from cv2 import cv2
from PIL import Image
import pandas as pd
from sklearn import svm


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
    return np.array(imgs),image_list
    

def ScatFeature(x_train):
    scat_x_train = Scattering2D(J=2, L=8)(x_train)
    final_scat_x_train = Flatten()(scat_x_train)
    # print(final_scat_x_train.shape)
    stand_x_train = StandardScaler().fit_transform(final_scat_x_train) # normalizing the features
    return stand_x_train

def OneClassSVM_train(data,image_list):

# nu ratio of anomaly points
    clf = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
    clf.fit(data)
    pred = clf.predict(data)
    normal = data[pred == 1]
    abnormal = data[pred == -1]
    print(normal.shape)
    print(abnormal.shape)
    #plt.plot(normal[:, 0], normal[:, 1], 'bx')
    #plt.plot(abnormal[:, 0], abnormal[:, 1], 'ro')
    #plt.show()
    #remove abnormal data
    for i in range (len(pred)):
        if pred[i] == -1:
            os.remove(image_list[i])
    #  # print abnormal data
    # for i in range (len(pred)):
    #     if pred[i] == -1:
    #         print(image_list[i])
    
    
    
# print(normal)
# print(abnormal)
# print(normal.shape)
# print(abnormal.shape)
# plt.plot(normal[:, 0], normal[:, 1], 'bx')
# plt.plot(abnormal[:, 0], abnormal[:, 1], 'ro')plt.show()

if __name__ == '__main__':
    train_dir =  "/semiconductor2/train/defect/"
    data,image_list = get_files(train_dir)
    stand_x_train = ScatFeature(data)
    OneClassSVM_train(stand_x_train,image_list)
    #os.remove
    #print "after : %s" %os.listdir(train_dir)