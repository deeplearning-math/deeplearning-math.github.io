import keras
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import mnist
import os
import numpy as np
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

data = mnist.load_data(path="mnist.npz")

(x_train, y_train) = data[0]

(x_test, y_test) = data[1]

IMG_SIZE = 32

def resize(img_array):
    tmp = np.empty((img_array.shape[0], IMG_SIZE, IMG_SIZE))

    for i in range(len(img_array)):
        img = img_array[i].reshape(28, 28).astype('uint8')
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32')/255
        tmp[i] = img
        
    return tmp

train_x_resize = resize(x_train)

train_x_final = np.stack((train_x_resize,)*3, axis=-1)

train_x_final = preprocess_input(train_x_final)

vgg19 = VGG19(weights='imagenet', include_top=False)

train_features = vgg19.predict(np.array(train_x_final), batch_size=30, verbose=1)

train_features = np.reshape(train_features, (60000,512))

test_x_resize = resize(x_test)

test_x_final = np.stack((test_x_resize,)*3, axis=-1)

test_x_final = preprocess_input(test_x_final)

vgg19 = VGG19(weights='imagenet', include_top=False)

test_features = vgg19.predict(np.array(test_x_final), batch_size=30, verbose=1)

test_features = np.reshape(test_features, (10000,512))

train_features = preprocessing.scale(train_features)

test_features = preprocessing.scale(test_features)


# LDA

lda = LinearDiscriminantAnalysis()
lda.fit(train_features, y_train)
lda_predict = lda.predict(test_features)
lda_accuracy = 0
for i in range(10000):
	if lda_predict[i] == y_test[i]:
		lda_accuracy += 1/10000
print(lda_accuracy)

lda_train = lda.predict(train_features)
lda_loss = 0
for i in range(60000):
	if lda_train[i] == y_train[i]:
		lda_loss += 1/60000
print(lda_loss)

# Logistic

lr = LogisticRegression(random_state=0).fit(train_features, y_train)
lr_predict = lr.predict(test_features)
lr_accuracy = 0
for i in range(10000):
	if lr_predict[i] == y_test[i]:
		lr_accuracy += 1/10000
print(lr_accuracy)

lr_train = lr.predict(train_features)
lr_loss = 0
for i in range(60000):
	if lr_train[i] == y_train[i]:
		lr_loss += 1/60000
print(lr_loss)

# SVM

svma = svm.SVC()
svma.fit(train_features, y_train)
svma_predict = svma.predict(test_features)
svma_accuracy = 0
for i in range(10000):
	if svma_predict[i] == y_test[i]:
		svma_accuracy += 1/10000
print(svma_accuracy)

svma_train = svma.predict(train_features)
svma_loss = 0
for i in range(60000):
	if svma_train[i] == y_train[i]:
		svma_loss += 1/60000
print(svma_loss)


# Random Forest

rf = RandomForestClassifier(max_depth=9, random_state=0)
rf.fit(train_features, y_train)
rf_predict = rf.predict(test_features)
rf_accuracy = 0
for i in range(10000):
	if rf_predict[i] == y_test[i]:
		rf_accuracy += 1/10000
print(rf_accuracy)

rf_train = rf.predict(train_features)
rf_loss = 0
for i in range(60000):
	if rf_train[i] == y_train[i]:
		rf_loss += 1/60000
print(rf_loss)



