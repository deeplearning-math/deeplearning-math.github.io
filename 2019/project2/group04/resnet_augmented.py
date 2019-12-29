#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:09:17 2016
@author: stephen
"""

#https://arxiv.org/abs/1705.09914

from tensorflow import keras
import numpy as np
import pandas as pd

np.random.seed(813306)
 
def build_resnet(input_shape, n_feature_maps, nb_classes):
    print ('build conv_x')
    x = keras.layers.Input(shape=(input_shape))
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv2D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, 1, 1,padding='same')(x)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1,padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1,padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1,padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1,padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    full = keras.layers.GlobalAveragePooling2D()(y)
    full = keras.layers.Flatten()(full)
    out = keras.layers.Dense(1, activation='linear')(full) #nb_classes, softmax
    print ('        -- model was built.')
    return x, out
   
nb_epochs = 1500
 
 
#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

# from sklearn.preprocessing import MinMaxScaler

# def myround(x, base=50):
#     return int(base * round(float(x)/base))


# path = 'C:/Users/cyril/Desktop/MPhil/Theories of ML/Project2/'
# name =  ['tails_train.csv','val.csv']
# data = pd.read_csv(path+name[0])
# data = data.drop(['Unnamed: 0'],axis=1)#,'index','time'
# target_train = data.target
# #data.target = np.where(data.target>0,1,0)
# data.target =  np.where(data.target>0.005,1,(np.where(data.target< -0.005,2, 0)))

# data_test = pd.read_csv(path+name[1])
# data_test= data_test.drop(['Unnamed: 0'],axis=1) #,'index','time'
# target = data_test.target
# data_test.target =  np.where(data_test.target>0.005,1,(np.where(data_test.target< -0.005,2, 0)))
# #data.columns = ['ret','first','second','third']
# #data_test.columns = ['ret','first','second','third']
# # get training, val and test set
# params ={'batch_size':50,'num_historical_days':1}
# trainSet = data
# testSet = data_test
# num_historical_days = int(params['num_historical_days'])
# scaler1 = MinMaxScaler(feature_range=(-1, 1)).fit(trainSet.iloc[:,1:])
# #scaler2 = StandardScaler().fit(np.array(trainSet.iloc[:,0]).reshape(-1,1))
# st_train= scaler1.transform(trainSet.iloc[:,1:])
# st_test= scaler1.transform(testSet.iloc[:,1:])
# st_train_days = []
# for i in range(num_historical_days,len(st_train)):
#   st_train_days.append(st_train[(i-int(num_historical_days)):i])
  
# #print(f'st_train {len(st_train_days)}')
# st_test_days = []
# for i in range(num_historical_days,len(st_test)):
#   st_test_days.append(st_test[(i-int(num_historical_days)):i])

# trainIndex =  myround(len(st_train_days),int(params['batch_size'])) - int(params['batch_size'])
# st_train_days = st_train_days[:trainIndex]
# #print(f'st_train {len(st_train_days)}')
# testIndex =  myround(len(st_test_days),int(params['batch_size'])) - int(params['batch_size'])
# st_test_days = st_test_days[:testIndex]    


# # Reshape variables for LSTM
# X_train = np.array(st_train_days)
# print(f'X_train.shape {X_train.shape}')
# X_train = X_train.reshape(X_train.shape + (1,))
# #y_train= np.array(scaler2.transform(np.array(trainSet.iloc[:,0]).reshape(-1,1))).reshape(len(st_train),)
# y_train = np.array(data.target[num_historical_days:trainIndex+num_historical_days]).reshape(X_train.shape[0],1)
# target_train = np.array(target_train[num_historical_days:trainIndex+num_historical_days])
# X_test = np.array(st_test_days)
# X_test = X_test.reshape(X_test.shape + (1,))
# y_test = np.array(testSet.iloc[num_historical_days:testIndex+num_historical_days,0]).reshape(X_test.shape[0],1)
# target = np.array(target[num_historical_days:testIndex+num_historical_days])
# nb_classes = 3
     
# x , y = build_resnet(X_train.shape[1:], 64, nb_classes)
# model = keras.models.Model(inputs=x, outputs=y)
# print(model.summary())
# optimizer = keras.optimizers.Adam()
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])
  
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
#                   patience=5, min_lr=0.0001) 
# hist = model.fit(X_train, y_train, batch_size=256,epochs=2,
#     #sample_weight=target_train**2+1e-16,
#     #validation_data=(X_test, y_test),  #,target**2+1e-16
#     verbose=1, callbacks = [reduce_lr])


# yhat = model.predict(X_test,batch_size=int(params['batch_size']))

# inv_yhat = np.array(yhat)
# lin = np.linspace(0,1,51)
# print(pd.DataFrame(inv_yhat).quantile(lin))
# arg_max = []
# for i in inv_yhat:
#   arg_max.append(np.argmax(i))
# arg_max = np.array(arg_max)
# arg_max = np.where(arg_max==1,1,(np.where(arg_max==2,-1,0)))
# #acc = accuracy_score(np.sign(arg_max),np.sign(y_test))#testSet['label']
# #mae = mean_squared_error(inv_yhat,inv_y)
# pd.DataFrame({'inv_yhat0':inv_yhat[:,1],'inv_yhat1':inv_yhat[:,2],'inv_y':target}).to_csv(path + "cnn.csv")
# #pickle.dump(params, open( path + "lstmBaselineTmp.p", "wb" ) )
# #print(cor)
# #print(acc)
