import struct,os
import numpy as np
import keras
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import pickle as pkl

### data loading start ###
def loadfeature (file):
    fr=open(file,'rb')
    inf = pickle.load(fr)
    inf = np.array(inf)
    # print(np.shape(inf))
    return inf
### data loading end ###


with open("{}.pkl".format('pretrained_flatten/VGG/test_vgg_flatten'),"rb") as file:
    test_image=pkl.load(file)
with open("{}.pkl".format('pretrained_flatten/VGG/test_vgg_label'),"rb") as file:
    test_label=pkl.load(file)
with open("{}.pkl".format('pretrained_flatten/VGG/train_vgg_flatten'),"rb") as file:
    train_image=pkl.load(file)    
    
with open("{}.pkl".format('pretrained_flatten/VGG/train_vgg_label'),"rb") as file:
    train_label=pkl.load(file)    


(train_image,  train_label), (test_image, test_label) = keras.datasets.mnist.load_data()
train_image=np.array([i.flatten() for i in train_image])
test_image=np.array([i.flatten() for i in test_image])


#main
if __name__=="__main__":
#    train_image = loadfeature('train_vgg_flatten.pkl')
#    train_label = loadfeature("train_vgg_label.pkl")
#    test_image = loadfeature("test_vgg_flatten.pkl")
#    test_label = loadfeature("test_vgg_label.pkl") #load data
     ###random forest start###
     rfc = RandomForestClassifier(n_estimators=100)
     rfc.fit(train_image, train_label)
     predict = rfc.predict(test_image)
     print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
     print("Classification report for classifier %s:\n%s\n" % (rfc, classification_report(test_label, predict)))
     ###random forest end###

     ###logistic regression start###
     lr = LogisticRegression()
     lr.fit(train_image, train_label)
     predict = lr.predict(test_image)
     print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
     print("Classification report for classifier %s:\n%s\n" % (lr, classification_report(test_label, predict)))
     ###logistic regression end###

     ###SVM start###
     svc = SVC()
     svc.fit(train_image,train_label)
     predict = svc.predict(test_image)
     print("accuracy_score: %.4lf" % accuracy_score(predict,test_label))
     print("Classification report for classifier %s:\n%s\n" % (svc, classification_report(test_label, predict)))
     ###SVM end###

     ###LDA start###
     lda = LinearDiscriminantAnalysis()
     lda.fit(train_image, train_label)
     predict = lda.predict(test_image)
     print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
     print("Classification report for classifier %s:\n%s\n" % (lda, classification_report(test_label, predict)))
     ##LDA end###

