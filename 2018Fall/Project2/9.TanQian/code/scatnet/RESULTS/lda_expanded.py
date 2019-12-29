import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# dataset = pd.read_csv('Wine.csv')
# X = dataset.iloc[:, 0:13].values
# y = dataset.iloc[:, 13].values

datasize =45200
a=np.loadtxt('feature_clear.csv',delimiter=',')
b=np.loadtxt('label_clear.csv',delimiter=',')

unknown_img=np.loadtxt('feature_unknown.csv',delimiter=',')



a_test=np.loadtxt('feature_test.csv',delimiter=',')
b_test=np.loadtxt('label_test.csv',delimiter=',')




X00=a
y00=b
a_test_N=a_test
#X00 = np.load('cifar_features2.npy')
#y00 = np.load('cifar_labels2.npy')
y0 = y00[0:datasize].reshape(datasize,1)
X0 = 1/np.amax(X00)*X00
a_test_NN=1/np.amax(X00)*a_test_N
unknown_img_NN=1/np.amax(X00)*unknown_img

X = np.zeros((datasize,391))
y = np.zeros((datasize,1))
perm_index = np.random.permutation(datasize)
for i in range(datasize):
    X[i] = X0[perm_index[i]]
    y[i] = y0[perm_index[i]]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
a_test_NN=sc.transform(a_test_NN)
unknown_img_NN=sc.transform(unknown_img_NN)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
a_test_NN = lda.transform(a_test_NN)
unknown_img_NN=lda.transform(unknown_img_NN)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
# from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
print('train predict:',classifier.predict(X_set[:20]))
print('train real  :',y_set[:20].reshape(1,20))
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue','cyan','yellow','purple','brown','orange','pink','gray')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green', 'blue','cyan','yellow','purple','brown','orange','pink','gray'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('LD1')
# plt.ylabel('LD2')
# plt.legend()
# plt.show()

# Visualising the Test set results
# from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
print('test predict:',classifier.predict(X_set[:20]))
print('test_real  :',y_set[:20].reshape(1,20))
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue','cyan','yellow','purple','brown','orange','pink','gray')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green', 'blue','cyan','yellow','purple','brown','orange','pink','gray'))(i), label = j)
# plt.title('Logistic Regression (Test set)')
# plt.xlabel('LD1')
# plt.ylabel('LD2')
# plt.legend()
# plt.show()

train_accuracy1=np.sum(np.abs(classifier.predict(X_train).reshape(40680,1)-y_train)==0)/40680
test_accuracy2=np.sum(np.abs(classifier.predict(X_test).reshape(4520,1)-y_test)==0)/4520
pred_disp=classifier.predict(a_test_NN)
unknown_label=classifier.predict(unknown_img_NN)

test_accuracy3=np.sum(np.abs(pred_disp-b_test)==0)/5000
bad_accuracy=sum(b_test*pred_disp)/sum(b_test)
good_accuracy=sum((b_test==0)*(pred_disp==0))/sum(b_test==0)

print(train_accuracy1)
print(test_accuracy2)
print(test_accuracy3)
print(bad_accuracy)
print(good_accuracy)

np.savetxt('label_lda_expanded_pre/label_unknown.csv',unknown_label,delimiter=',')
#0.801819075712881
#0.7918141592920354
#0.8232
#0.7221052631578947
#0.8338121546961326