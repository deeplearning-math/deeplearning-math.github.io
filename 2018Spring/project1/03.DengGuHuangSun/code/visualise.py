from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

N_components = 2
N_neighbors = 10
N_jobs = 6

def colour(l):
    colours = []
    for i in l:
        if i==1: colours.append('xkcd:blue')
        else: colours.append('xkcd:red')
    return colours

def make_plot(dfs, labels, methods, n_plots=6, s=4, title=None):
    r_flags = np.array([False] * len(labels), dtype=np.bool)
    nr_flags = np.array([False] * len(labels), dtype=np.bool)
    for i in range(len(labels)):
        if labels[i] == 1: r_flags[i] = True
        else: nr_flags[i] = True
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(8,8), dpi=80)
        x = dfs
        x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
        x = (x - x_min) / (x_max - x_min)
        ax.scatter(x[:,0], x[:,1], c=colours, s=s)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(methods.upper())
    else:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(32,16), dpi=100)
        for (x, ax, method) in zip(dfs, np.reshape(axes,(-1)), methods):
            x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
            x = (x - x_min) / (x_max - x_min)
            for I,J,K in [(r_flags, 'Raphael', 'xkcd:blue'), (nr_flags, 'Non-Raphael', 'xkcd:red')]:
                xs = x[I, 0]
                ys = x[I, 1]
                ax.scatter(xs, ys, s=s, c=K, label=J)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(method.upper(), fontsize=24)
            #ax.set_facecolor('grey')
    if title is not None:
        plt.suptitle(title, fontsize=36)
    hdls, lbls = ax.get_legend_handles_labels()
    fig.legend(hdls, lbls, loc='right', fontsize=20, markerscale=4)
    plt.savefig('vis_res/results.png')

def visualize(data, method, n_jobs=1, n_neighbors=N_neighbors, n_components=N_components):
    print('Computing {} embeddings ... '.format(method.upper()))
    t0 = time()
    if method == 'Pca':
        out = decomposition.TruncatedSVD(n_components=n_components).fit_transform(data)
    elif method == 'Mds':
        out = manifold.MDS(n_components=n_components,
            n_init=1, max_iter=100, n_jobs=n_jobs).fit_transform(data)
    elif method == 'Isomap':
        out = manifold.Isomap(n_components=n_components, n_jobs=n_jobs).fit_transform(data)
    elif method == 'Spectral Embedding':
        out = manifold.SpectralEmbedding(n_components=n_components,
            n_neighbors=n_neighbors, n_jobs=n_jobs).fit_transform(data)
    elif method == 'TSNE':
        out = manifold.TSNE(
            n_components=n_components, init='pca', random_state=0).fit_transform(data)
    else:
        out = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
            eigen_solver='auto', method=method.split(' ')[0], n_jobs=n_jobs).fit_transform(data)
    print('{} finished. Time elapsed: {:f}'.format(method.upper(), time()-t0))
    return out

if __name__ == '__main__':
    methods = ['Pca', 'Mds', \
        'Isomap', 'Spectral Embedding', 'TSNE', 'standard LLE']
    labels = np.loadtxt('labels4200.txt')
    #print(labels)
    features = np.loadtxt('res_feats.txt')
    #features = np.random.randn(20, 6)
    #features[:,-1] = np.random.randint(2, size=20)
    labels = features[:,-1]

    for i in methods:
        df = visualize(data=features[:,:-1], method=i, n_jobs=N_jobs)
        np.savetxt('vis_res/{}.txt'.format(i), df)

    dfs = []
    for i in methods:
        dfs.append(np.loadtxt('vis_res/{}.txt'.format(i)))
        make_plot(dfs, labels, methods, title='Resnet-18 Features Visualisation', n_plots=6)
