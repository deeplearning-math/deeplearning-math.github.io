import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skmetrics
from sklearn.decomposition import PCA

X=np.load("./Train/train.npy")
y=np.load("./Train/label.npy")


X1=X[:7100,:]
y1=y[:7100]


p1= PCA(n_components=60)
newX = p1.fit_transform(X1)

#x_train,x_test,y_train,y_test=train_test_split(X1,y1,test_size=0.2,random_state=1)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#KNN
folds=10
k_choices=[1,3,5,7,9,11,13,15]#k近邻
X_folds=[]
Y_folds=[]
X_folds=np.vsplit(newX,folds)#将x_train纵向等分为4个片段
Y_folds=np.hsplit(y1,folds)#将y_train横向等分为4个片段

accuracy_of_k={}#字典，储存不同k的准确率
for k in k_choices:
    accuracy_of_k[k]=[]#每个k的准确率
    

for i in range(folds):
    X_train=np.vstack(X_folds[:i]+X_folds[i+1:])#交叉验证中的训练集
    X_val=X_folds[i]#交叉验证中的测试集
    Y_train=np.hstack(Y_folds[:i]+Y_folds[i+1:])#交叉验证中的训练集
    Y_val=Y_folds[i]#交叉验证中的测试集
    #print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)
    
    for k in k_choices:
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,Y_train)
        Y_val_pred=knn.predict(X_val)
        accuracy=skmetrics.accuracy_score(Y_val_pred, Y_val)
        accuracy_of_k[k].append(accuracy)#对应的k中加入准确率
    
for k in sorted(k_choices):
#    for accuracy in accuracy_of_k[k]:
    print("k=%d,accuracy=%f"%(k,sum(accuracy_of_k[k])/folds))

model = KNeighborsClassifier(n_neighbors=3)
model.fit(newX,y1)
X_pred= model.predict(newX)


t=np.load("./Test/test26.npy")

t = p1.transform(t)
#t1=p1.inverse_transform(t) 
#print(pca.explained_variance_ratio_)
predicted = model.predict(t)

c0=0
c1=0
for i in predicted:
    if(i==0):
        c0=c0+1
    if(i==1):
        c1=c1+1
print(c0,c1, 100*c1/(c0+c1))

print('Accuracy score :', skmetrics.accuracy_score(X_pred, y1))


#LR
folds=10
X_folds=[]
Y_folds=[]
X_folds=np.vsplit(X1,folds)#将x_train纵向等分为4个片段
Y_folds=np.hsplit(y1,folds)#将y_train横向等分为4个片段

accuracy=[]

   
for i in range(folds):
    X_train=np.vstack(X_folds[:i]+X_folds[i+1:])#交叉验证中的训练集
    X_val=X_folds[i]#交叉验证中的测试集
    Y_train=np.hstack(Y_folds[:i]+Y_folds[i+1:])#交叉验证中的训练集
    Y_val=Y_folds[i]#交叉验证中的测试集

    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, Y_train)
    lr.predict( X_val)
    score = lr.score( X_val, Y_val)
    accuracy.append(score)

print(sum(accuracy)/10)



model= LogisticRegression(solver='liblinear')
model.fit(X1, y1)
print(lr.score(X1, y1))

t=np.load("./Test/23n.npy")
predicted = model.predict(t)

c0=0
c1=0
for i in predicted:
    if(i==0):
        c0=c0+1
    if(i==1):
        c1=c1+1
print(c0,c1, 100*c1/(c0+c1))

print('Accuracy score :', skmetrics.accuracy_score(X_pred, y1))

