import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import torch
### Load dataset
from torchvision.datasets import MNIST
from resnet_vis import *
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
#os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
#os.system('rm tmp')

mnist_train = MNIST(download=True, train=True, root=".").train_data.float()
data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist_train.mean()/255,), (mnist_train.std()/255,))])    
mnist_dataset_train = MNIST(download=True, root=".", transform=data_transform, train=True)

mnist_dataset_test = MNIST(download=True, train=False, root=".", transform=data_transform)

#### Load ResNet 18 model to get features
resnet_model = resnet18Conv5().cuda()


### Generate train features

train_num = 1200
batch_size = 400
len_train = int(train_num // batch_size)
train_input = [None]*len_train
train_label = np.zeros((train_num,))
for i in range(len_train):
    print("Load features for train data %05d/%05d"%(i, len_train))
    batch_data = [None]*batch_size
    for k in range(i*batch_size, (i+1)*batch_size):
        data = mnist_dataset_train[k]
        X = data[0].cuda()
        X = X.unsqueeze(0)#1x28x28
        batch_data[k - i*batch_size] = X
        train_label[k] = data[1]
    batch_data_cat = torch.cat([batch_data[p] for p in range(batch_size)], dim=0)
    outputs = resnet_model(batch_data_cat)
    outputs = outputs.detach().cpu().numpy()
    feature_vec = np.reshape(outputs, [batch_size, -1])
    train_input[i] = feature_vec
    

train_input_data = np.concatenate([train_input[k] for k in range(len_train)], axis=0)


### Generate test features
test_num = 1200
batch_size = 400
len_test = int(test_num // batch_size)
test_input = [None]*len_test
test_label = np.zeros((test_num,))
for i in range(len_test):
    print("Load features for test data %05d/%05d"%(i, len_test))
    batch_data = [None]*batch_size
    for k in range(i*batch_size, (i+1)*batch_size):
        data = mnist_dataset_test[k]
        X = data[0].cuda()
        X = X.unsqueeze(0)#1x28x28
        batch_data[k - i*batch_size] = X
        test_label[k] = data[1]
    batch_data_cat = torch.cat([batch_data[p] for p in range(batch_size)], dim=0)
    outputs = resnet_model(batch_data_cat)
    outputs = outputs.detach().cpu().numpy()
    feature_vec = np.reshape(outputs, [batch_size, -1])
    test_input[i] = feature_vec

test_input_data = np.concatenate([test_input[k] for k in range(len_test)], axis=0)

# Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)

classifier = OneVsRestClassifier(svm.SVC(gamma=0.001), n_jobs=10)
classifier.fit(train_input_data, train_label)

print("training finished")


predicted = classifier.predict(test_input_data)

print("test finished")

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_label, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label, predicted))
