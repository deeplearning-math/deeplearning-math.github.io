# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:42:01 2019

Four traditional supervised learning method.
Use sklearn to do the PCA, RandomForest, KNN, Logistic Regression
plot the ROC curve of four methods

@author: KL
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression



X = train    # import train data, Here train is a numpy array extracted by VGG16.
Y = label    # import the label of train data. It's also a numpy array.


pca = PCA(n_components=100)
X_pca = pca.fit(X).transform(X)
print(np.sum(pca.explained_variance_ratio_))


L1=[]
L=[0,10,25,50,75,100,125,150,175,200,300,400,500]
for i in L:
    p1=PCA(n_components=i)
    X1_pca = p1.fit(X).transform(X)
    print(np.sum(p1.explained_variance_ratio_))
    L1.append(np.sum(p1.explained_variance_ratio_))
    
#plt.scatter(L, L1)
plt.plot(L, L1)
plt.xlabel('Principal Components')
plt.ylabel('Explained variance ratio')
plt.show()

#kmeans1 = KMeans(n_clusters=2, random_state=0).fit(X_pca)
#y_1 = kmeans1.labels_
#Kmeans_mis_number1 = np.abs((Y-y_1)).sum()

X_train = X_pca[:5000,:]
Y_train = label[:5000]
X_test = X_pca[5000:,:]
Y_test = label[5000:]

rfc = RandomForestClassifier(n_estimators=50, random_state=42)
rfc.fit(X_train, Y_train)
rfc_disp = plot_roc_curve(rfc, X_test, Y_test)
plt.show()

logi = LogisticRegression(random_state=0,max_iter=150)
logi.fit(X_train, Y_train)

logi_disp = plot_roc_curve(logi, X_test, Y_test)
plt.show()

n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X_train, Y_train)
clf_disp = plot_roc_curve(clf, X_test, Y_test)
plt.show()

clf_score = clf.score(X_test,Y_test)

svc = SVC(random_state=42)
svc.fit(X_train, Y_train)
ax = plt.gca()
svc_disp = plot_roc_curve(svc, X_test, Y_test, ax=ax, alpha=0.8)
clf_disp.plot(ax=ax, alpha=0.6)
rfc_disp.plot(ax=ax, alpha=0.4)
logi_disp.plot(ax=ax, alpha=0.9)
plt.show()

svc_score = svc.score(X_test,Y_test)
clf_score = clf.score(X_test,Y_test)
rfc_score = rfc.score(X_test,Y_test)
logi_score = logi.score(X_test,Y_test)

print(svc_score)
print(clf_score)
print(rfc_score)
print(logi_score)






T=[]
Test=np.load(r"D:\zhenghui\P\Test\test\test26.npy")
T.append(Test)
T1=[]
for i in T:
    T1.append(pca.transform(i))

R=[]
for i in T1:
    R.append(svc.predict(i))
 
svc=[]    
for j in R:
    c0=0
    c1=0
    for i in j:
        if(i==0):
            c0=c0+1
        if(i==1):
            c1=c1+1
    print(c0,c1,100*c1/(c0+c1))
    svc.append(100*c1/(c0+c1))


L=[1,2,3,4,5,6,7]
plt.plot(L, svc, label='SVC',marker='o')
plt.plot(L, clf, label='KNeignborsClassifier',marker='o')
plt.plot(L, logi, label='LogisticRegression',marker='o')
plt.plot(L, rf, label='RandomForestClassifier',marker='o')
plt.xlabel('Image Number')
plt.ylabel('Possibility')
plt.legend()
plt.show()


