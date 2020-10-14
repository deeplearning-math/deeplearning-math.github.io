import keras
import tensorflow as tf
import cv2 as cv
import numpy as np
import pickle as pkl
from keras.applications.resnet import ResNet50
from keras.models import Model
import numpy as np
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = [cv.cvtColor(cv.resize(i, (32, 32)), cv.COLOR_GRAY2RGB)
           for i in x_train
           ]

x_train = np.concatenate([i[np.newaxis]
                          for i in x_train
                          ]).astype(np.float32)

x_test = [cv.cvtColor(cv.resize(i, (32,32)), cv.COLOR_GRAY2RGB)
          for i in x_test
          ]
x_test = np.concatenate([i[np.newaxis]
                         for i in x_test
                         ]).astype(np.float32)

x_train = x_train / 255
x_test = x_test / 255
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)
model = ResNet50(weights='imagenet', pooling=max, include_top = False)

resnet_features = model.predict(x_train)
print(resnet_features.shape)

resnet_features2 = model.predict(x_test)
print(resnet_features2.shape)

with open("{}.pkl".format('train_resnet'),"wb") as file:
        pkl.dump(resnet_features,file)

with open("{}.pkl".format('train_resnet_label'),"wb") as file:
        pkl.dump(y_train,file)        

with open("{}.pkl".format('test_resnet'),"wb") as file:
        pkl.dump(resnet_features2,file)

with open("{}.pkl".format('test_resnet_label'),"wb") as file:
        pkl.dump(y_test,file)        
        
        

with open("{}.pkl".format('train_resnet'),"rb") as file:
    test=pkl.load(file)
    vgg_train_flatten=np.array([i.flatten() for i in train_vgg ])
    
with open("{}.pkl".format('train_resnet_flatten'),"wb") as file:
        pkl.dump(test2,file)

