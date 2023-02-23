from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np

print("randomnum", np.random.rand())
X, y = datasets.load_iris(return_X_y=True)
print("X: ", X)
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
print("clusters:", kmeans.labels_)


