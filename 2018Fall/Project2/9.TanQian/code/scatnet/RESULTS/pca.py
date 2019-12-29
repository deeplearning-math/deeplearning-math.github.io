import numpy as np
from sklearn.decomposition import PCA
fea=np.loadtxt('feature.csv',delimiter=',')
feap=fea
X = feap[0:5000,:].reshape(5000,391)
X=(X-np.mean(X))/np.std(X)
#feap=np.zeros((28,391))
#for i in range(28):
#    feap[i,:]=fea[i*3,:];
#X = feap[:,:].reshape(28,391)
X_embedded = PCA(n_components=2).fit_transform(X)
print(X_embedded.shape)
np.savetxt('2_feature.csv',X_embedded,delimiter=',')
 