import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
from kymatio.torch import Scattering2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

scattering = Scattering2D(J=2, shape=(28, 28)).cuda().eval()
train_data = torchvision.datasets.FashionMNIST(
    ".",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)
data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=4096 * 3,
                                          shuffle=False)
tmp = []
with torch.no_grad():
    for batch in data_loader:
        t = scattering(batch[0].cuda())
        # t = batch[0]
        tmp.append(t.cpu())
train_x = torch.cat(tmp).numpy()
train_x = train_x.reshape((train_x.shape[0], -1))
train_y = train_data.targets.type(torch.FloatTensor).numpy()
print("Finished loading train data")

test_data = torchvision.datasets.FashionMNIST(
    ".",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)
data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=4096 * 3,
                                          shuffle=False)
tmp = []
with torch.no_grad():
    for batch in data_loader:
        t = scattering(batch[0].cuda())
        # t = batch[0]
        tmp.append(t.cpu())
test_x = torch.cat(tmp).numpy()
test_x = test_x.reshape((test_x.shape[0], -1))
test_y = test_data.targets.type(torch.FloatTensor).numpy()
print("Finished loading test data")

if False:
    ### First PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(train_x)
    print('Explained variation per principal component: {}'.format(
        pca.explained_variance_ratio_))
    plt.figure()
    sns.scatterplot(x=pca_result[:, 0],
                    y=pca_result[:, 1],
                    hue=train_y,
                    palette=sns.color_palette("hls", 10),
                    legend="full",
                    alpha=0.3)
    plt.title('PCA Clusters')
    plt.show(block=True)

    # Do a PCA over 50 dims fist then TSNE
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(train_x)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result)
    plt.figure()
    sns.scatterplot(x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    hue=train_y,
                    palette=sns.color_palette("hls", 10),
                    legend="full",
                    alpha=0.3)
    plt.title('t-SNE Clusters')
    plt.show(block=True)

if True:
    # calculate feature statistics
    num_class = int(train_y.max()) + 1
    mu_G = train_x.mean(axis=0, keepdims=True)

    mu_c = []
    for i in range(num_class):
        c = float(i)
        train_x_c = train_x[np.where(train_y == c)[0], :]
        mu_c.append(train_x_c.mean(axis=0, keepdims=True))
    mu_c = np.concatenate(mu_c, axis=0)

    tmp = train_x - mu_G
    sigma_T = tmp.T @ tmp / train_x.shape[0]

    tmp = mu_c - mu_G
    sigma_B = tmp.T @ tmp / float(num_class)

    tmp = train_x.copy()
    for i in range(tmp.shape[0]):
        tmp[i, :] -= mu_c[int(train_y[i]), :]
    sigma_W = tmp.T @ tmp / train_x.shape[0]
    sigma_W = sigma_W.astype(np.double)
    sigma_T = sigma_T.astype(np.double)
    sigma_B = sigma_B.astype(np.double)
    print(sigma_B.max(), sigma_B.min())
    print(sigma_T.max(), sigma_T.min())
    print(sigma_W.max(), sigma_W.min())

    diff = np.abs((sigma_T - sigma_B - sigma_W))
    print("diff max: ", diff.max())
    print("diff mean: ", diff.mean())

    NC1 = np.trace(sigma_W @ np.linalg.pinv(sigma_B)) / float(num_class)
    print("NC1: ", NC1)

    eq_norm = mu_c - mu_G
    eq_norm = np.sqrt(np.square(eq_norm).sum(axis=1))
    eq_norm = eq_norm.std() / eq_norm.mean()
    print("equal norms: ", eq_norm)

    mu_c_centered = mu_c - mu_G
    norms = np.sqrt(np.square(mu_c_centered).sum(axis=1))
    pairwise_dot = []
    for i in range(mu_c_centered.shape[0]):
        for j in range(mu_c_centered.shape[0]):
            if i == j:
                continue
            res = (mu_c_centered[i, :] *
                   mu_c_centered[j, :]).sum() / norms[i] / norms[j]
            pairwise_dot.append(res)
    pairwise_dot = np.array(pairwise_dot)
    print("equal angle: ", pairwise_dot.std())
    print("maximal angle equiangularity: ",
          np.abs(pairwise_dot + 1.0 / (float(num_class) - 1.0)).mean())

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

file_name = 'scatter-net/lr.sav'
if os.path.exists(file_name):
    lr = pickle.load(open(file_name, 'rb'))
else:
    lr = SGDClassifier(loss="log",
                       penalty="l2",
                       max_iter=1000,
                       eta0=0.1,
                       verbose=0)
    lr.fit(train_x, train_y)
    pickle.dump(lr, open(file_name, 'wb'))

file_name = 'scatter-net/rf.sav'
if os.path.exists(file_name):
    rf = pickle.load(open(file_name, 'rb'))
else:
    rf = RandomForestClassifier(n_estimators=100, verbose=0)
    rf.fit(train_x, train_y)
    pickle.dump(rf, open(file_name, 'wb'))

file_name = 'scatter-net/lda.sav'
if os.path.exists(file_name):
    lda = pickle.load(open(file_name, 'rb'))
else:
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_x, train_y)
    pickle.dump(lda, open(file_name, 'wb'))

yhat = lr.predict(train_x)
print("Result for logistic regression ******************")
print(classification_report(train_y, yhat))
print("acc=" + str((yhat == train_y).astype('float').sum() / len(train_y)))

yhat = rf.predict(train_x)
print("Result for random forest ******************")
print(classification_report(train_y, yhat))
print("acc=" + str((yhat == train_y).astype('float').sum() / len(train_y)))

yhat = lda.predict(train_x)
print("Result for LDA ******************")
print(classification_report(train_y, yhat))
print("acc=" + str((yhat == train_y).astype('float').sum() / len(train_y)))
