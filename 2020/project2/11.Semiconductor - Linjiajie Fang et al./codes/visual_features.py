from abc import ABC
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from model import ResNet
from data_processing import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

DefectDetector = ResNet(1, 1).to(device)
DefectDetector.load_state_dict(torch.load('DefectDetector', map_location='cpu'))
DefectDetector.eval()

"""
sample data
"""

tensor, label = boostrap_training_data(100, shuffle=False)

extracted_feature = DefectDetector.activation_vector(tensor)


"""
loading from GPU
"""

extracted_feature = torch.load('feature', map_location=torch.device('cpu'))
label = torch.load('label', map_location=torch.device('cpu'))

"""
tSNE
"""

train_features_embedded = TSNE(n_components=2).fit_transform(extracted_feature.detach().numpy())


feature_data = pd.DataFrame(train_features_embedded, columns=['factor_1', 'factor_2'])
feature_data['label'] = label.int()
fig, axe = plt.subplots(1, 1, figsize=(10, 10))
fig.suptitle(f"t-SNE plot of the activation vectors from 32-dim space to 2-dim space ", fontsize=13)
sns.scatterplot(data=feature_data, x="factor_1", y="factor_2", hue="label", palette="deep", ax=axe)
fig.show()


"""
PCA
"""
pca = PCA(n_components=2)
train_features_embedded = pca.fit_transform(extracted_feature.detach().numpy())

feature_data = pd.DataFrame(train_features_embedded, columns=['factor_1', 'factor_2'])
feature_data['label'] = label.int()
fig, axe = plt.subplots(1, 1, figsize=(10, 10))
fig.suptitle(f"PCA plot of the activation vectors from 32-dim space to 2-dim space ", fontsize=13)
sns.scatterplot(data=feature_data, x="factor_1", y="factor_2", hue="label", palette="deep", ax=axe)
fig.show()




