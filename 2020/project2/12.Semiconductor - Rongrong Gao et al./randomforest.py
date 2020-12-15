import torchvision.models as models
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd

import pickle
from scipy import signal, misc, ndimage
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


original_model = models.resnext101_32x8d(pretrained=True)
original_model = torch.nn.DataParallel(original_model)
checkpoint = torch.load('models_para/resnext101_32x8_focal/model_best.pth.tar')
original_model.load_state_dict(checkpoint['state_dict'])


class resnextConv5(nn.Module):
    def __init__(self):
        super(resnextConv5, self).__init__()

        self.features = nn.Sequential(
            *list(original_model.children())[:-3]
        )
    def forward(self, x):
        x = self.features(x)
        return x

resnet_model = resnextConv5().cuda()
batch_size=1
traindir = '../input/semi-conductor-image-classification-second-stage/train/train_contest'
valdir = '../input/semi-conductor-image-classification-second-stage/train/validation'
testdir = '../input/semi-conductor-image-classification-second-stage/test/test_contest'
normalize = transforms.Normalize(mean=[0.2297, 0.2297, 0.2297],
                                     std=[0.1508, 0.1508, 0.1508])


train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
        ]))
train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)


train_num = len(train_dataset)
print(train_num)
train_input = [None]*train_num
train_label = np.zeros((train_num,))

for i, (images, target) in enumerate(train_loader):
    if i%500==0:
        print("Load features for train data %05d/%05d"%(i, train_num))

    outputs = resnet_model(images.cuda())
    outputs = outputs.detach().cpu().numpy()
    feature_vec = np.reshape(outputs, [batch_size, -1])
    # print(feature_vec.shape[1])
    train_input[i] = feature_vec
    train_label[i] = target[0]
    

train_features = np.concatenate([train_input[k] for k in range(train_num)], axis=0)
print(train_features.shape)
clf = RandomForestClassifier(n_estimators=100, n_jobs=10,)
clf.fit(train_features,train_label)

print("training finished")

with open('FINETUNED_focal2.pickle','wb') as f:
	pickle.dump(clf, f)



#val

val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
        ]))
val_loader = torch.utils.data.DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
print(val_dataset.class_to_idx)

val_num = len(val_dataset)
print(val_num)
val_input = [None]*val_num
val_label = np.zeros((val_num,))

for i, (images, target) in enumerate(val_loader):
    if i%500==0:
        print("Load features for train data %05d/%05d"%(i, val_num))

    outputs = resnet_model(images.cuda())
    outputs = outputs.detach().cpu().numpy()
    feature_vec = np.reshape(outputs, [batch_size, -1])
    if feature_vec.shape[1]!=200704:
        print(i)
    val_input[i] = feature_vec
    val_label[i] = target[0]
    

val_features = np.concatenate([val_input[k] for k in range(val_num)], axis=0)
print(val_features.shape)

pickle_in = open('MNIST_RFC1.pickle','rb')
clf = pickle.load(pickle_in)


# predicted = clf.predict(val_features)

acc = clf.predict_proba(val_features)
print(acc)
print(val_label)


#test
test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
        ]))
test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

## Generate test features

test_num = len(test_dataset)
print(test_num)
test_input = [None]*test_num
test_label = np.zeros((test_num,))

for i, (images, _ ) in enumerate(test_loader):
    if i%500==0:
        print("Load features for train data %05d/%05d"%(i, test_num))

    outputs = resnet_model(images.cuda())
    outputs = outputs.detach().cpu().numpy()
    feature_vec = np.reshape(outputs, [batch_size, -1])
    test_input[i] = feature_vec
    

test_features = np.concatenate([test_input[k] for k in range(test_num)], axis=0)
print(test_features.shape)


pickle_in = open('FINETUNED_focal2.pickle','rb')
clf = pickle.load(pickle_in)

confidences = clf.predict_proba(test_features)

img_names = [img[0].split("/")[-1][:-4] for img in test_dataset.imgs]
df_submit = pd.DataFrame(img_names, columns=['id'])


confidences_np = confidences[:,0]
df_submit['defect_score'] = confidences_np
df_submit.to_csv('test_rf_finedfocalnew.csv', index=False)