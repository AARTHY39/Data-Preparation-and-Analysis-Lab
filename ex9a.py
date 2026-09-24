import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
iris = load_iris()
X = iris.data
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
n_clusters = 3
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_labels = agg_clustering.fit_predict(X)
plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=agg_labels,
    cmap='rainbow',
    s=50
)
plt.title("Agglomerative Clustering on Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()