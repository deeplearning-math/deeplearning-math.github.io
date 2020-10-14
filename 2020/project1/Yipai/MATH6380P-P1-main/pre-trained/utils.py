import os
import copy
from torchvision import datasets
from torchvision import transforms
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.data.dataset import TensorDataset
from time import time
import time
import torch
from joblib import dump, load
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import torch.optim as optim
from torch.optim import lr_scheduler
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_with_labels(weights, labels):
    plt.cla()

    X, Y = weights[:, 0], weights[:, 1]
    for s in range(0, 10):
        c = cm.rainbow(int(255 * s / 9))
        plt.scatter(X[labels == s],
                    Y[labels == s],
                    s=10,
                    color=c,
                    alpha=.8,
                    lw=1,
                    label=str(s))

    plt.xlim(X.min() - 10, X.max() + 10)
    plt.ylim(Y.min() - 10, Y.max() + 10)
    plt.legend(loc='lower right', shadow=False, scatterpoints=1)


def visualize_TSNE(features, labels):
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(features)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result)
    plt.figure()
    sns.scatterplot(x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    hue=labels,
                    palette=sns.color_palette("hls", 10),
                    legend="full",
                    alpha=0.3)
    plt.title('t-SNE Clusters')
    plt.show(block=True)


def visualize_PCA(features, labels):
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(features.numpy())
    plt.figure()
    sns.scatterplot(x=embeddings[:, 0],
                    y=embeddings[:, 1],
                    hue=labels,
                    palette=sns.color_palette("hls", 10),
                    legend="full",
                    alpha=0.3)
    plt.title("PCA Clusters")


def visualize_LDA(features, labels):
    lda = LinearDiscriminantAnalysis(n_components=2)
    plot_only = 500
    embeddings = lda.fit_transform(features.numpy()[:plot_only, :],
                                   labels.numpy()[:plot_only])
    labels = labels[:plot_only]
    plot_with_labels(embeddings, labels)
    plt.title("LDA")


class TransformedMNIST:
    def __init__(self):
        super(TransformedMNIST, self).__init__()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset_transforms = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(), normalize
        ])
        self.train_dataset = datasets.FashionMNIST(
            os.getcwd() + "/data",
            train=True,
            transform=dataset_transforms,
            download=True)
        self.test_dataset = datasets.FashionMNIST(os.getcwd() + "/data",
                                                  train=False,
                                                  transform=dataset_transforms,
                                                  download=True)

    def get_train(self):
        return (self.train_dataset)

    def get_test(self):
        return (self.test_dataset)


def imshow(inp, title=None, normalize=True):
    inp = inp.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean

    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, interpolation="bilinear", aspect="auto")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


class vgg19extractor(nn.Module):
    def __init__(self):
        super(vgg19extractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = torch.nn.Sequential(*list(vgg19.classifier[:-1]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return (x)


class resnetextractor(torch.nn.Module):
    def __init__(self):
        super(resnetextractor, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class FeatureExtractor:
    def __init__(self, model):

        self.model = model
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad: bool = False

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def features(self,
                 dataloader,
                 save_to_disk=True,
                 train=True,
                 flatten_config=None):
        feat_coll = []
        label_coll = []

        for batch_id, [features, labels] in enumerate(dataloader):

            print("Batch {}, features shape: {}, labels shape: {}".format(
                batch_id, features.shape, labels.shape))
            features = features.to(self.device)
            labels = labels.to(self.device)

            t1 = time()
            out = self.model(features)
            t2 = time()
            print("Output shape: {}, Time taken: {}".format(
                out.shape, t2 - t1))
            out = out.to("cpu")
            features = features.to("cpu")

            if flatten_config is not None:
                try:
                    start_dim = flatten_config["start_dim"]
                except KeyError:
                    start_dim = 0

                try:
                    end_dim = flatten_config["end_dim"]
                except KeyError:
                    end_dim = -1

                out = torch.flatten(out, start_dim=start_dim, end_dim=end_dim)
                print("Flattend output shape: {}".format(out.shape))

            feat_coll.append(out)
            label_coll.append(labels)

        out_features = torch.flatten(torch.stack(feat_coll),
                                     start_dim=0,
                                     end_dim=1)
        out_labels = torch.flatten(torch.stack(label_coll),
                                   start_dim=0,
                                   end_dim=1)

        print("The final features matrix has shape: {}".format(
            out_features.shape))

        if save_to_disk:

            out_dataset = TensorDataset(out_features, out_labels)
            if train:
                prefix = "train"
            else:
                prefix = "test"
            filename = "{}_{}_dataset.pt".format(prefix,
                                                 self.model.__class__.__name__)
            torch.save(out_dataset, filename)
            print("Saved features at {}/{}".format(os.getcwd(), filename))

        return out_features, out_labels


def extract_features(save=True, train=True):
    models = []
    vgg19 = vgg19extractor()
    resnet = resnetextractor()
    models.append(vgg19)
    models.append(resnet)
    dataset = TransformedMNIST()
    batch_size = 100
    for model in models:
        name = model.__class__.__name__
        extractor = FeatureExtractor(model)
        data = dataset.get_train() if train else dataset.get_test()
        dataloader = DataLoader(data, batch_size=batch_size)
        _ = extractor.features(dataloader, save, train)


def train_data_dict():

    return {
        "vgg19":
        "/Users/yongquanqu/Desktop/pre_trained/train_vgg19extractor_dataset.pt",
        "resnet":
        "/Users/yongquanqu/Desktop/pre_trained/train_resnetextractor_dataset.pt"
    }


def get_train_data(name):
    all_data = train_data_dict()

    return all_data[name]


def test_data_dict():

    return {
        "vgg19":
        "/Users/yongquanqu/Desktop/pre_trained/test_vgg19extractor_dataset.pt",
        "resnet":
        "/Users/yongquanqu/Desktop/pre_trained/test_resnetextractor_dataset.pt"
    }


def get_test_data(name):
    all_data = train_data_dict()

    return all_data[name]


def get_stored_dataset(dataset, train=True):
    datasets = train_data_dict() if train else test_data_dict()
    loaded_dataset = torch.load(datasets[dataset],
                                map_location=torch.device('cpu'))
    features = loaded_dataset[:][0]
    labels = loaded_dataset[:][1]

    return features, labels


def get_stored_model(model_name, dataset, cv=False):
    dataset_dir = "/Users/yongquanqu/Desktop/pre_trained/"
    filename = "{}_{}_{}.joblib".format(model_name, "cv" if cv else "nocv",
                                        dataset)
    classifier = load(dataset_dir + filename)

    return classifier


def train_classifier(name, dataset="vgg19", cv=False, save=True):
    lr = SGDClassifier(loss="log",
                       penalty="l2",
                       max_iter=1000,
                       eta0=0.1,
                       verbose=5)

    rf = RandomForestClassifier(n_estimators=100, verbose=5)

    lda = LinearDiscriminantAnalysis()

    classifiers = {"lr": lr, "lda": lda, "rf": rf}

    param_grid = dict()
    param_grid = dict()
    param_grid["lr"] = {"penalty": ["l2", "l1", "elasticnet"]}
    param_grid["rf"] = {"n_estimators": [10, 100, 500]}

    scaler = preprocessing.StandardScaler()

    classifier = classifiers[name]
    train_features, train_labels = get_stored_dataset(dataset, train=True)
    train_features = train_features.numpy()
    train_labels = train_labels.numpy()
    scaler.fit(train_features)

    classifier_to_fit = classifier if not cv else GridSearchCV(
        classifier, param_grid[name], verbose=5)

    classifier_to_fit.fit(scaler.transform(train_features), train_labels)

    if save:
        filename = "{}_{}_{}.joblib".format(name, "cv" if cv else "nocv",
                                            dataset)
        dump(classifier_to_fit,
             "/Users/yongquanqu/Desktop/pre_trained" + filename)

    return classifier_to_fit


def test_classifier(name, dataset=["vgg19"], cv=False):
    filename = "{}_{}_{}.joblib".format(name, "cv" if cv else "nocv", dataset)
    classifier = load("/Users/yongquanqu/Desktop/pre_trained/" + filename)
    train_features, _ = get_stored_dataset(dataset, train=True)
    train_features = train_features.numpy()
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_features)

    test_features, _ = get_stored_dataset(dataset, train=False)
    test_features = test_features.numpy()

    preds = classifier.predict(scaler.transform(test_features))

    return preds
