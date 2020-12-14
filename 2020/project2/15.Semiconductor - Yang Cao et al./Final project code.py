!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir .kaggle
import json
token = {"username":"jiaminwu4","key":"4dab3de4edb4b77efadf3c53f28639fb"}
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)
!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
!kaggle competitions download -c semi-conductor-image-classification-second-stage -p /content/project2/
!unzip '/content/project2/semi-conductor-image-classification-second-stage.zip' -d '/content/project2'
print('finished unzip')
!pip install kymatio
import torch, torchvision
import sys, os
import matplotlib.pyplot as plt
import numpy as np
#from kymatio.torch import HarmonicScattering3D
from torchvision import transforms, datasets
from matplotlib import cm
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import keras
import time
data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
flip_axes = lambda tens: tens.permute(1, 2, 0)
semicond_dataset = datasets.ImageFolder(root='/content/project2/train/train_contest/',
                                        transform=transforms.Compose(
                                            [
                                                transforms.Resize([267, 275]),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]
                                        )
                                       )
dataset_loader = torch.utils.data.DataLoader(semicond_dataset,
                                             batch_size=2027, shuffle=True,
                                             num_workers=0)
batch_id, (train_images, train_labels) = next(enumerate(dataset_loader))
train_images.shape
dataset_loader_origi = torch.utils.data.DataLoader(semicond_dataset,
                                             batch_size=500, shuffle=True,
                                             num_workers=0)
batch_id, (train_images_origi, train_labels_origi) = next(enumerate(dataset_loader_origi))
train_images_array=torch.flatten(train_images_origi,start_dim=1).cpu().detach().numpy()
train_labels_array=train_labels_origi.cpu().detach().numpy()
print(train_images_array.shape)
data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
flip_axes = lambda tens: tens.permute(1, 2, 0)
semicond_dataset_test = datasets.ImageFolder(root='/content/project2/test/test_contest/',
                                        transform=transforms.Compose(
                                            [
                                                transforms.Resize([267, 275]),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]
                                        )
                                       )
dataset_loader_test = torch.utils.data.DataLoader(semicond_dataset_test,
                                             batch_size=10, shuffle=False,
                                             num_workers=0)
batch_id, (test_images, test_labels) = next(enumerate(dataset_loader_test))
test_images_array=torch.flatten(test_images,start_dim=1).cpu().detach().numpy()
test_labels_array=test_labels.cpu().detach().numpy()
test_images.shape
scattering = HarmonicScattering3D(J=2, shape=(3, 267, 275), L=2)
scattering.cuda()
features_proce = []
labels_proce = []

for batch_id, [train_images, train_labels] in enumerate(dataset_loader):
    print("Batch {}, features shape: {}, labels shape: {}".format(batch_id, train_images.shape, train_labels.shape))
    train_images = train_images.cuda()

    features_scat_proce = scattering(train_images)


    features_scat_proce = torch.flatten(features_scat_proce, start_dim=1)
    print("Flattend feature shape: {}".format(features_scat_proce.shape))

    features_proce.append(features_scat_proce)
    labels_proce.append(train_labels)

features_scattering = torch.flatten(torch.stack(features_proce), start_dim=0, end_dim=1)
labels_scattering = torch.flatten(torch.stack(labels_proce), start_dim=0, end_dim=1)

print("The final features matrix has shape: {}".format(features_scattering.shape))
print("The final labels matrix has shape: {}".format(labels_scattering.shape))
scattering = HarmonicScattering3D(J=2, shape=(3, 267, 275), L=2)
scattering.cuda()
features_proce_test = []
labels_proce_test = []

for batch_id, [test_images, test_labels] in enumerate(dataset_loader_test):

    print("Batch {}, features shape: {}, labels shape: {}".format(batch_id, test_images.shape, test_labels.shape))
    test_images = test_images.cuda()

    features_scat_proce_test = scattering(test_images)


    features_scat_proce_test = torch.flatten(features_scat_proce_test, start_dim=1)
    print("Flattend feature shape: {}".format(features_scat_proce_test.shape))

    features_proce_test.append(features_scat_proce_test)
    labels_proce_test.append(test_labels)

features_scattering_test = torch.flatten(torch.stack(features_proce_test), start_dim=0, end_dim=1)
labels_scattering_test = torch.flatten(torch.stack(labels_proce_test), start_dim=0, end_dim=1)

print("The final features matrix has shape: {}".format(features_scattering_test.shape))
print("The final labels matrix has shape: {}".format(labels_scattering_test.shape))
out_dataset = TensorDataset(features_scattering, labels_scattering)
filename='scattering_features.pt'
torch.save(out_dataset, filename)
print("Saved features at {}/{}".format(os.getcwd(), filename))

out_dataset_test = TensorDataset(features_scattering_test, labels_scattering_test)
filename_test='scattering_features_test.pt'
torch.save(out_dataset_test, filename_test)
print("Saved features at {}/{}".format(os.getcwd(), filename_test))
def get_stored_dataset(filename, train=True):
    loaded_dataset = torch.load(filename)
    features = loaded_dataset[:][0]
    labels = loaded_dataset[:][1]

    return features, labels
filename = 'scattering_features.pt'
features_scattering, labels_scattering = get_stored_dataset(filename)
filename_test = 'scattering_features_test.pt'
features_scattering_test, labels_scattering_test = get_stored_dataset(filename_test)
print(features_scattering.shape)
print(features_scattering_test.shape)
features_scattering_array=features_scattering.cpu().detach().numpy()
labels_scattering_array=labels_scattering.cpu().detach().numpy()

print(features_scattering_array.shape)
features_scattering_array_test=features_scattering_test.cpu().detach().numpy()
labels_scattering_array_test=labels_scattering_test.cpu().detach().numpy()

print(features_scattering_array_test.shape)


# !pip install scikit-cuda
def visualization(visual_method, feature_mehtod):
    feature_method = feature_mehtod
    visual_method = visual_method
    if feature_method == 'raw data':
        data = train_images_array[0:500, :]
        data_labels = train_labels_array[0:500]
    if feature_method == 'scattering net':
        data = features_scattering_array[0:500, :]
        data_labels = labels_scattering_array[0:500]
    if feature_method == 'resnet50':
        data = features_resnet50_array[0:1000, :]
        data_labels = labels_resnet50_array[0:1000]
    # Visualize
    print('start visualization')
    if visual_method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        result = pca.fit_transform(data)

    if visual_method == 'tsne':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0)
        result = tsne.fit_transform(data)

    if visual_method == 'mds':
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, random_state=0)
        result = mds.fit_transform(data)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    classes = np.sort(np.unique(data_labels))
    colors = ['red', 'blue']
    labels = ["good", "bad"]
    for class_ix, color, label in zip(
            classes, colors, labels):
        ax.scatter(result[np.where(data_labels == class_ix), 0],
                   result[np.where(data_labels == class_ix), 1],
                   color=color, edgecolor='whitesmoke',
                   linewidth='1', alpha=0.9, label=label)
        ax.legend(loc='best')
    plt.title('{} on {}'.format(visual_method, feature_mehtod))
    print(visual_method)
    print(feature_mehtod)
    plt.savefig('{} on {}.png'.format(visual_method, feature_mehtod), format='png')
    plt.show()


# visualization('pca', 'raw data')
visualization('tsne', 'raw data')
# visualization('mds', 'raw data')
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

def classification(test_image, test_label, class_method, feature_method):
  feature_method=feature_method
  class_method=class_method
  if feature_method == 'raw data':
    train_image=train_images_array
    train_label=train_labels_array
  if feature_method == 'scattering net':
    train_image=features_scattering_array
    train_label=labels_scattering_array
  if feature_method == 'resnet50':
    train_image=features_resnet50_array
    train_label=labels_resnet50_array
###random forest start###
  if class_method == 'rf':
     print('start rf')
     rfc = RandomForestClassifier(n_estimators=100)
     trained_model = rfc.fit(train_image, train_label)
     trained_model.fit(train_image, train_label)
     predict = rfc.predict(test_image)
     train_error=1-accuracy_score(train_label, trained_model.predict(train_image))
     test_error=1-accuracy_score(predict, test_label)
     print("train error for {} on {}: {}".format(class_method, feature_method, train_error))
     print("test error for {} on {}: {}".format(class_method, feature_method, test_error))
     ###random forest end###

###logistic regression start###
  if class_method == 'lr':
     print('start lr')
     lr = LogisticRegression()
     trained_model = lr.fit(train_image, train_label)
     trained_model.fit(train_image, train_label)
     predict = lr.predict(test_image)
     train_error=1-accuracy_score(train_label, trained_model.predict(train_image))
     test_error=1-accuracy_score(predict, test_label)
     print("train error for {} on {}: {}".format(class_method, feature_method, train_error))
     print("test error for {} on {}: {}".format(class_method, feature_method, test_error))
     ###logistic regression end###

###svm start###
  if class_method == 'svm':
     print('start svm')
     svc = SVC()
     trained_model = svc.fit(train_image, train_label)
     trained_model.fit(train_image, train_label)
     predict = svc.predict(test_image)
     train_error=1-accuracy_score(train_label, trained_model.predict(train_image))
     test_error=1-accuracy_score(predict, test_label)
     print("train error for {} on {}: {}".format(class_method, feature_method, train_error))
     print("test error for {} on {}: {}".format(class_method, feature_method, test_error))
     ###svm end###

###lda start###
  if class_method == 'lda':
     print('start lda')
     lda = LinearDiscriminantAnalysis()
     trained_model = lda.fit(train_image, train_label)
     trained_model.fit(train_image, train_label)
     predict = lda.predict(test_image)
     train_error=1-accuracy_score(train_label, trained_model.predict(train_image))
     test_error=1-accuracy_score(predict, test_label)
     print("train error for {} on {}: {}".format(class_method, feature_method, train_error))
     print("test error for {} on {}: {}".format(class_method, feature_method, test_error))
     ###lda end###
#classification(test_images_array, test_labels_array, 'rf', 'raw data')
#classification(test_images_array, test_labels_array, 'lr', 'raw data')
#classification(test_images_array, test_labels_array, 'svm', 'raw data')
classification(test_images_array, test_labels_array, 'lda', 'raw data')