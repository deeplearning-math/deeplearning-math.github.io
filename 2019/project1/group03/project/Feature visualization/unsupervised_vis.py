# This scripy do 2 components and 3 components PCA,t-SNE,LDA(supervised_method) for 
# original image, ResNet features and Scattering features
# author: jiaxin xie
# Download all the data from  https://drive.google.com/drive/folders/1NyCbi_9o4tW7xtFjONmo5ZXfpJXmeIC-?usp=sharing
# t-SNE will take a long time.



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.manifold import TSNE
#from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#colors
targets = [0,1,2,3,4,5,6,7,8,9]
colors = cm.rainbow(np.linspace(0, 1, len(targets)))

#load labels
data_y = np.load('scatter_labels.npy')

#load sampe_index
sample_index=np.load('samples.npy')

'''
#load original_image
mnist_data = pd.read_csv("mnist_train.csv",header=None)
new_column_name = ['label']
for num in range(mnist_data.shape[1]-1):
       tem = 'pixel' + str(num)
       new_column_name.append(tem)
mnist_data.columns = new_column_name
data_intensity=mnist_data.drop(["label"], axis = 1).values
data_intensity=data_intensity[sample_index]

# original image TSNE2
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(data_intensity)
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('original_image 2 components TSNE', fontsize = 20)


for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_tsne[indicesToKeep, 0]
   , x_tsne[indicesToKeep, 1]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

#original image PCA2
pca = PCA(n_components=2)
x_pca = pca.fit_transform(data_intensity)
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('original_image 2 components PCA', fontsize = 20)


for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_pca[indicesToKeep, 0]
   , x_pca[indicesToKeep, 1]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


# original image TSNE3
tsne = TSNE(n_components=3)
x_tsne = tsne.fit_transform(data_intensity)

fig = plt.figure( )
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_zlabel('Component 3', fontsize = 15)
ax.set_title('original_image 3 components TSNE', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_tsne[indicesToKeep, 0]
   , x_tsne[indicesToKeep, 1]
   , x_tsne[indicesToKeep,2]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
'''

# resnet features
features=np.load('resnet_feature.npy')
data_x = features

#resnet LDA2
lda = LDA(n_components=2)
x_lda = lda.fit_transform(data_x, data_y)
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('resnet 2 components LDA', fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_lda[indicesToKeep, 0]
   , x_lda[indicesToKeep, 1]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


pca_std = StandardScaler().fit_transform(data_x)

'''
# resnet PCA3
pca = PCA(n_components=3)
PA3 = pca.fit_transform(pca_std)

fig = plt.figure( )
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('resnet 3 components PCA', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(PA3[indicesToKeep, 0]
   , PA3[indicesToKeep, 1]
   , PA3[indicesToKeep,2]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
'''

#resnet PCA2
pca = PCA(n_components=2)
PA2= pca.fit_transform(pca_std)
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('resnet 2 components PCA', fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(PA2[indicesToKeep, 0]
   , PA2[indicesToKeep, 1]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# resnet TSNE 2
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(data_x)
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('resnet 2 components TSNE', fontsize = 20)


for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_tsne[indicesToKeep, 0]
   , x_tsne[indicesToKeep, 1]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

'''
#resnet TSNE 3
tsne = TSNE(n_components=3)
x_tsne = tsne.fit_transform(data_x)
fig = plt.figure( )
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_zlabel('Component 3', fontsize = 15)
ax.set_title('resnet 3 components TSNE', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_tsne[indicesToKeep, 0]
   , x_tsne[indicesToKeep, 1]
   , x_tsne[indicesToKeep,2]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
'''


# scatter features
features=np.load('scatter_feature.npy')
data_x = features
lda = LDA(n_components=2)
x_lda = lda.fit_transform(data_x, data_y)
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('scatter 2 components LDA', fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_lda[indicesToKeep, 0]
   , x_lda[indicesToKeep, 1]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


pca_std = StandardScaler().fit_transform(data_x)
'''
# scatter PCA3
pca = PCA(n_components=3)
PA3 = pca.fit_transform(pca_std)

fig = plt.figure( )
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('scatter 3 components PCA', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(PA3[indicesToKeep, 0]
   , PA3[indicesToKeep, 1]
   , PA3[indicesToKeep,2]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
'''

#scatter PCA2
pca = PCA(n_components=2)
PA2= pca.fit_transform(pca_std)
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('scatter 2 components PCA', fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(PA2[indicesToKeep, 0]
   , PA2[indicesToKeep, 1]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# scatter TSNE 2
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(data_x)
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('scatter 2 components TSNE', fontsize = 20)


for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_tsne[indicesToKeep, 0]
   , x_tsne[indicesToKeep, 1]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

'''
#scatter TSNE 3
tsne = TSNE(n_components=3)
x_tsne = tsne.fit_transform(data_x)
fig = plt.figure( )
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_zlabel('Component 3', fontsize = 15)
ax.set_title('scatter 3 components TSNE', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = [i for i,x in enumerate(data_y) if x==target]
    ax.scatter(x_tsne[indicesToKeep, 0]
   , x_tsne[indicesToKeep, 1]
   , x_tsne[indicesToKeep,2]
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
'''

