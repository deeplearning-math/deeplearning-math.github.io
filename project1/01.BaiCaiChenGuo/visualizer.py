from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.datasets import fetch_mldata

def vis_pca(feature):
    ''' visualize extracted features using Principal Component Analysis '''
    if len(feature.shape) == 5:
        feature = feature.reshape(feature.shape[:-4] +  (-1 , feature.shape[3], feature.shape[4]))
        feature = feature.reshape((feature.shape[0], -1))
    elif len(feature.shape) == 4:
        feature = feature.reshape((feature.shape[0], -1))
    pca =  PCA(2)
    projected = pca.fit_transform(feature)
    return projected
    

def vis_tsne(feature):
    ''' visualize extracted features using t-distributed Stochastic Neighbor Embedding '''
    tsne = TSNE(n_components = 2, perplexity=40, verbose=0)
    if len(feature.shape) == 5:
        feature = feature.reshape(feature.shape[:-4] +  (-1 , feature.shape[3], feature.shape[4]))
        feature = feature.reshape((feature.shape[0], -1))
    elif len(feature.shape) == 4:
        feature = feature.reshape((feature.shape[0], -1))
    projected = tsne.fit_transform(feature)
    return projected
    

def vis_mds(feature):
    ''' visualize extracted features using Multidimensional Scaling '''
    if len(feature.shape) == 5:
        feature = feature.reshape(feature.shape[:-4] +  (-1 , feature.shape[3], feature.shape[4]))
        feature = feature.reshape((feature.shape[0], -1))
    elif len(feature.shape) == 4:
        feature = feature.reshape((feature.shape[0], -1))
    mds = MDS(n_components = 2, max_iter=100, n_init=1)
    projected = mds.fit_transform(feature)
    return projected

def vis_lle(feature):
    ''' visualize extracted features using Local Linear Embedding '''
    if len(feature.shape) == 5:
        feature = feature.reshape(feature.shape[:-4] +  (-1 , feature.shape[3], feature.shape[4]))
        feature = feature.reshape((feature.shape[0], -1))
    elif len(feature.shape) == 4:
        feature = feature.reshape((feature.shape[0], -1))
    lle = LocallyLinearEmbedding(n_neighbors = 7, n_components = 2,
                                     eigen_solver='auto', method='standard')
    projected = lle.fit_transform(feature)
    return projected
    

def vis_isomap(feature):
    ''' visualize extracted features using Isometric Mapping '''
    if len(feature.shape) == 5:
        feature = feature.reshape(feature.shape[:-4] +  (-1 , feature.shape[3], feature.shape[4]))
        feature = feature.reshape((feature.shape[0], -1))
    elif len(feature.shape) == 4:
        feature = feature.reshape((feature.shape[0], -1))
    isomap = Isomap(n_neighbors = 7, n_components = 2)
    projected = isomap.fit_transform(feature)
    return projected

def vis_se(feature):
    ''' visualize extracted features using Spectral Embedding '''
    if len(feature.shape) == 5:
        feature = feature.reshape(feature.shape[:-4] +  (-1 , feature.shape[3], feature.shape[4]))
        feature = feature.reshape((feature.shape[0], -1))
    elif len(feature.shape) == 4:
        feature = feature.reshape((feature.shape[0], -1))
    se = SpectralEmbedding(n_neighbors = 7, n_components = 2)
    projected = se.fit_transform(feature)
    return projected
