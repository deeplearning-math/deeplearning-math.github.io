import torch
import torchvision
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from kymatio.scattering2d.filter_bank import filter_bank
from kymatio.scattering2d.utils import fft2
from kymatio.torch import Scattering2D
from sklearn.manifold import TSNE

data_path = '/Users/fanglinjiajie/locals/datasets/'

TRANS = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
BATCH_SIZE = 100

train_set = torchvision.datasets.MNIST(data_path, train=True, transform=TRANS, target_transform=None, download=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.MNIST(data_path, train=False, transform=TRANS, target_transform=None, download=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


"""
Dataset visualization
"""
def imshow(img):
    #  img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

idx = []
num = list(range(10))
for i in range(len(labels)):
    if labels[i] in num:
        idx.append(i)
        num.remove(labels[i])
    if not num:
        break

img = images[idx]

# show images
imshow(torchvision.utils.make_grid(img, nrow=10))
# print labels


"""
plot scattering functions
"""

M = 28
J = 2
L = 8
filters_set = filter_bank(M, M, J, L=L)


def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 / (1.0 + abs(z[idx]) ** 0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c


fig, axs = plt.subplots(J, L, sharex=True, sharey=True)
fig.set_figheight(4)
fig.set_figwidth(6)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
i = 0
for filter in filters_set['psi']:
    f = filter[0]
    filter_c = fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    axs[i // L, i % L].imshow(colorize(filter_c))
    axs[i // L, i % L].axis('off')
    axs[i // L, i % L].set_title(
        "$j = {}$ \n $\\theta={}$".format(i // L, i % L))
    i = i + 1

fig.suptitle((r"Wavelets for each scales $j$ and angles $\theta$ used."
              "\nColor saturation and color hue respectively denote complex"
              "\nmagnitude and complex phase."), fontsize=13)
fig.show()

plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.axis('off')
plt.set_cmap('gray_r')
f = filters_set['phi'][0]
filter_c = fft2(f)
filter_c = np.fft.fftshift(filter_c)
plt.suptitle("The corresponding low-pass filter (scaling function)", fontsize=13)
filter_c = np.abs(filter_c)
plt.imshow(filter_c)

plt.show()


"""
plot the transformed examples
"""
img = images[:8]
# show images
imshow(torchvision.utils.make_grid(img))
# print labels


scattering = Scattering2D(J=2, shape=(28, 28), max_order=1)
sx = scattering(img)

imshow(torchvision.utils.make_grid((sx[:, :, 0, :, :] * 255).int(), nrow=10))
for i in range(1, 17):
    imshow(torchvision.utils.make_grid((4 * sx[:, :, i, :, :] * 255).int(), nrow=10))


"""
plot training curves
"""
with open('record.pickle', 'rb') as rec:
    record = pickle.load(rec)

record = pd.DataFrame(record)
record = record.rename(columns={'nc11': 'Within-class_variation',
                                'nc12': 'Std/Avg',
                                'angle1': 'Std_of_pairwise_cosine_measure',
                                'angle2': 'Maximal-angle_equi-angularity'})


fig, axe = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle(r'Training the scattering net')
sns.lineplot(data=record[['training_loss', 'test_loss']], ax=axe[0])
sns.lineplot(data=record[['training_accuracy', 'test_accuracy']], ax=axe[1])
axe[0].set_xlabel('Epoch')
axe[0].set_ylabel('Loss')
axe[1].set_xlabel('Epoch')
axe[1].set_ylabel('Accuracy')
fig.show()

fig, axe = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle(r'Validation of neural collapse phenomena')
sns.lineplot(data=record[['Within-class_variation', 'Std/Avg']], ax=axe[0])
sns.lineplot(data=record[['Std_of_pairwise_cosine_measure', 'Maximal-angle_equi-angularity']], ax=axe[1])
axe[0].set_xlabel('Epoch')
axe[1].set_xlabel('Epoch')
axe[0].axvline(x=40, c='r')
axe[0].annotate(xy=(45, 5), text='0 Error', c='r')
axe[1].axvline(x=40, c='r')
fig.show()


"""
t-SNE plotting of the scattering&deep features
"""
from ScatteringNet import Scattering2dResNet

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Scattering2dResNet(17, 2).to(device)
model.eval()
model.load_state_dict(torch.load('ScatterNet_MNIST', map_location=torch.device('cpu')))
scattering = Scattering2D(J=2, shape=(28, 28), max_order=1)


def tsne_plot_features(extrac_func):
    def func(name=''):
        train_features = torch.tensor([])
        targets = torch.tensor([])

        for _ in range(5):
            x, y = iter(train_loader).next()
            train_features = torch.cat((train_features, extrac_func(x)), dim=0)
            targets = torch.cat((targets, y), dim=0)

        train_features = train_features.data.view(len(train_features), -1).numpy()
        train_features_embedded = TSNE(n_components=2).fit_transform(train_features)

        feature_data = pd.DataFrame(train_features_embedded, columns=['factor_1', 'factor_2'])
        feature_data['lable'] = targets.int()

        fig, axe = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f"t-SNE plot of the activation vectors from {train_features.shape[1]}-dim space to 2-dim space "
                     f"({name})", fontsize=13)
        sns.scatterplot(data=feature_data, x="factor_1", y="factor_2", hue="lable", palette="deep", ax=axe)
        fig.show()

    return func


@tsne_plot_features
def scatter_plot(x):
    return scattering(x)


@tsne_plot_features
def ScatNet_plot(x):
    model(scattering(x))
    return model.act_vec


resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
resnet.eval()


@tsne_plot_features
def ResNet_plot(x):
    # change to 2 channels!
    return resnet(torch.cat([x, x, x], dim=1))


vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
vgg.eval()


@tsne_plot_features
def vgg_plot(x):
    """
    require resize of the image to at least 244*244
    TRANS = torchvision.transforms.Compose([torchvision.transforms.Resize(244), torchvision.transforms.ToTensor()])
    """
    # change to 2 channels!
    return vgg(torch.cat([x, x, x], dim=1))


scatter_plot('Scattering')
ScatNet_plot('ScatNet')
ResNet_plot('ResNet18')
vgg_plot('VGG11')


"""
other traditional learning methods
"""
from sklearn.linear_model import LogisticRegression as logit
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as adb

img_data = test_set.data.view(-1, 1, 28, 28).float()/255
img_data = scattering(img_data).view(len(test_set), -1)
target_data = test_set.targets.cpu().numpy()

svc_clfer = svm.SVC(probability=True)
rf_clfer = rf(random_state=0)
adb_clfer = adb()
lg_clfer = logit()

for clfer in [svc_clfer, rf_clfer, adb_clfer, lg_clfer]:
    clfer.fit(img_data, target_data)


@tsne_plot_features
def svc_plot(x):
    x = scattering(x)
    x = x.view(-1, 17*7*7).numpy()
    return torch.tensor(svc_clfer.predict_proba(x))


@tsne_plot_features
def rf_plot(x):
    x = scattering(x)
    x = x.view(-1, 17*7*7).numpy()
    return torch.tensor(rf_clfer.predict_proba(x))


@tsne_plot_features
def adb_plot(x):
    x = scattering(x)
    x = x.view(-1, 17*7*7).numpy()
    return torch.tensor(adb_clfer.predict_proba(x))


@tsne_plot_features
def logit_plot(x):
    x = scattering(x)
    x = x.view(-1, 17*7*7).numpy()
    return torch.tensor(lg_clfer.predict_proba(x))


rf_plot('Random forest')
adb_plot('Adaboost')
svc_plot('SVM')
logit_plot('Logistic Regression')

