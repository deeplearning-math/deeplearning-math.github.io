#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:05:27 2018

@author: dengyizhe
"""
#Scattering:
x = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/2694_8688AC88-A1EF-4715-A39F-77F99D029569.npy")


x_test = x[x[:,416]==0]
x_train = x[x[:,416]!=0].astype(np.int32)

fea_train = x_train[:,:416]
fea_label = x_train[:,416].astype(np.int32)

###Logistic 
from sklearn import linear_model
from sklearn.model_selection import KFold
logreg = linear_model.LogisticRegression(penalty='l2',C=1)
kf = KFold(n_splits=21)
label=[]
True_label=[]
for train_index, val_index in kf.split(x_train):
    logreg.fit(fea_train[train_index],fea_label[train_index])
    a=logreg.predict(fea_train[val_index])
    b=1 if np.sum(a[a==1])>50 else -1
    True_label.append(fea_label[val_index][0])
    label.append(b)
label.reverse()
True_label.reverse()
sum(1  for i in range(21) if label[i]==True_label[i])


predictions=[]
predictions_test=logreg.predict(x_test[:,:-1])
for i in range(7):
    prediction=np.argmax(np.bincount(predictions_test[i*100:(i+1)*100]+1))-1
    predictions.append(prediction)
print(predictions)
####result:
#CV accuracy: 0.9523809523809523
#Prediction:[1, 1, -1, 1, 1, 1, 1]

######SVM
from sklearn import svm
from sklearn.model_selection import KFold

svc = svm.SVC(kernel='linear',C=1)
kf = KFold(n_splits=21)
label=[]
True_label=[]
for train_index, val_index in kf.split(x_train):
    svc.fit(fea_train[train_index],fea_label[train_index])
    a=svc.predict(fea_train[val_index])
    b=1 if np.sum(a[a==1])>50 else -1
    True_label.append(fea_label[val_index][0])
    label.append(b)
label.reverse()
True_label.reverse()
sum(1  for i in range(21) if label[i]==True_label[i])/21

predictions=[]
predictions_test=svc.predict(x_test[:,:-1])
for i in range(7):
    prediction=np.argmax(np.bincount(predictions_test[i*100:(i+1)*100]+1))-1
    predictions.append(prediction)
print(predictions)
####result:
#CV accuracy: 0.9047619047619048
#Prediction:[1, 1, -1, 1, 1, 1, 1]

#####KNN
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
svc = KNeighborsClassifier(6)
kf = KFold(n_splits=21)
label=[]
True_label=[]
for train_index, val_index in kf.split(x_train):
    svc.fit(fea_train[train_index],fea_label[train_index])
    a=svc.predict(fea_train[val_index])
    b=1 if np.sum(a[a==1])>50 else -1
    True_label.append(fea_label[val_index][0])
    label.append(b)
label.reverse()
True_label.reverse()
sum(1  for i in range(21) if label[i]==True_label[i])/21

predictions=[]
predictions_test=svc.predict(x_test[:,:-1])
for i in range(7):
    prediction=np.argmax(np.bincount(predictions_test[i*100:(i+1)*100]+1))-1
    predictions.append(prediction)
print(predictions)
55
[1, -1, -1, -1, -1, 1, -1]

######RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
clf = RandomForestClassifier(n_estimators=200)
kf = KFold(n_splits=21)
label=[]
True_label=[]
for train_index, val_index in kf.split(x_train):
    clf.fit(fea_train[train_index],fea_label[train_index])
    a=clf.predict(fea_train[val_index])
    b=1 if np.sum(a[a==1])>50 else -1
    True_label.append(fea_label[val_index][0])
    label.append(b)
label.reverse()
True_label.reverse()
sum(1  for i in range(21) if label[i]==True_label[i])/21
#71.42
#[1, -1, -1, 1, -1, -1, 1]
predictions=[]
predictions_test=clf.predict(x_test[:,:-1])
for i in range(7):
    prediction=np.argmax(np.bincount(predictions_test[i*100:(i+1)*100]+1))-1
    predictions.append(prediction)
print(predictions)


import numpy as np
import pandas as pd


x1 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/1. 185x354.TIF.features.npy")
x2 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/2. 202x216.TIF.features.npy")
x3 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/3. 272x369.TIF.features.npy")
x4 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/4. 363x189.tiff.features.npy")
x5 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/5. 379x281.tiff.features.npy")
x6 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/6. 197x168.tiff.features.npy")
x7 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/7. 243x413 .tiff.features.npy")
x8 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/8. 329x232.tif.features.npy")
x9 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/9. 279x187.tif.features.npy")
x10 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/10. 269x227.TIF.features.npy")
x11 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/11. 394x248.jpg.features.npy")
x12 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/12. 305x492.jpg.features.npy")
x13 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/13. 206x184.jpg.features.npy")
x14 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/14. 279x152.jpg.features.npy")
x15 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/15. 279x152.jpg.features.npy")
x16 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/16. 335x476.jpg.features.npy")
x17 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/17. 283x300.jpg.features.npy")
x18 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/18. 217x278.jpg.features.npy")
x19 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/19. 283x191.jpg.features.npy")
x20 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/20. 135x390.tif.features.npy")
x21 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/21. 203x258.jpg.features.npy")
x22 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/22. 257x375.jpg.features.npy")
x23 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/23. 84x68.tif.features.npy")
x24 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/24. 203x224.tif.features.npy")
x25 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/25. 156x115.tif.features.npy")
x26 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/26. 197x187.tif.features.npy")
x27 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/27. 278x419.tiff.features.npy")
x28 = np.load("/Users/dengyizhe/Desktop/MATH6380O/MiniProj/cropped_feats/28. 267x264.TIF.features.npy")
frames = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28]
x = x1
for i in range(27):
    x = np.vstack((x,frames[i+1]))

x=np.load()

x_test = x[x[:,-1]==0]
x_train = x[x[:,-1]!=0].astype(np.int32)

fea_train = x_train[:,:-1]
fea_label = x_train[:,-1].astype(np.int32)


#svm
from sklearn import svm
from sklearn.model_selection import KFold
svc = svm.SVC(kernel='linear',C=1)
kf = KFold(n_splits=21)
label=[]
True_label=[]
for train_index, val_index in kf.split(x_train):
    svc.fit(fea_train[train_index],fea_label[train_index])
    a=svc.predict(fea_train[val_index])
    b=1 if np.sum(a[a==1])>100 else -1
    True_label.append(fea_label[val_index][0])
    label.append(b)
label.reverse()
True_label.reverse()

sum(1  for i in range(21) if label[i]==True_label[i])/21


predictions=[]
predictions_test=svc.predict(x_test[:,:-1])
for i in range(7):
    prediction=np.argmax(np.bincount(predictions_test[i*200:(i+1)*200]+1))-1
    predictions.append(prediction)
print(predictions)

####result:
#CV accuracy: 80.95
#Prediction:[1, 1, -1, -1, -1, 1, -1]


#logistic Regression
from sklearn import linear_model
from sklearn.model_selection import KFold
logreg = linear_model.LogisticRegression(penalty='l2',C=1)
kf = KFold(n_splits=21)
label=[]
True_label=[]
for train_index, val_index in kf.split(x_train):
    logreg.fit(fea_train[train_index],fea_label[train_index])
    a=logreg.predict(fea_train[val_index])
    b=1 if np.sum(a[a==1])>100 else -1
    True_label.append(fea_label[val_index][0])
    label.append(b)
label.reverse()
True_label.reverse()
sum(1  for i in range(21) if label[i]==True_label[i])/21

predictions=[]
predictions_test=logreg.predict(x_test[:,:-1])
for i in range(7):
    prediction=np.argmax(np.bincount(predictions_test[i*200:(i+1)*200]+1))-1
    predictions.append(prediction)
print(predictions)

####result:
#CV accuracy: 0.8095
#Prediction:[1, -1, -1, -1, -1, 1, -1]
#knn 85.71 1 ---111

#RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
clf = RandomForestClassifier(n_estimators=300)
kf = KFold(n_splits=21)
label=[]
True_label=[]
for train_index, val_index in kf.split(x_train):
    clf.fit(fea_train[train_index],fea_label[train_index])
    a=clf.predict(fea_train[val_index])
    b=1 if np.sum(a[a==1])>100 else -1
    True_label.append(fea_label[val_index][0])
    label.append(b)
label.reverse()
True_label.reverse()
sum(1  for i in range(21) if label[i]==True_label[i])/21

predictions=[]
predictions_test=clf.predict(x_test[:,:-1])
for i in range(7):
    prediction=np.argmax(np.bincount(predictions_test[i*200:(i+1)*200]+1))-1
    predictions.append(prediction)
print(predictions)
####result:
#CV accuracy: 0.7619
#Prediction:[1, 1, -1, -1, 1, 1, -1]
