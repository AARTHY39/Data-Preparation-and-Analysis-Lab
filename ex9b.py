import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
X = np.array([[1, 2], [2, 2], [2, 3], [8, 8], [9, 8], [9, 9]])
def divisive_clustering(X, indices=None, next_cluster=None, Z=None):
    if indices is None:
        indices = list(range(len(X)))
    if Z is None:
        Z = []
    if next_cluster is None:
        next_cluster = len(X)
    if len(indices) <= 1:
        return Z, next_cluster, indices[0]
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X[indices])
    labels = kmeans.labels_
    cluster1 = [indices[i] for i in range(len(indices)) if labels[i] == 0]
    cluster2 = [indices[i] for i in range(len(indices)) if labels[i] == 1]
    Z, next_cluster, left = divisive_clustering(
        X, cluster1, next_cluster, Z
    )
    Z, next_cluster, right = divisive_clustering(
        X, cluster2, next_cluster, Z
    )
    dist = np.linalg.norm(
        X[cluster1].mean(axis=0) -
        X[cluster2].mean(axis=0)
    )
    Z.append([left, right, dist, len(indices)])
    current_cluster = next_cluster
    next_cluster += 1
    return Z, next_cluster, current_cluster
Z, _, _ = divisive_clustering(X)
Z = np.array(Z)
plt.figure(figsize=(6, 4))
dendrogram(Z, labels=list(range(len(X))))
plt.title("Divisive Hierarchical Clustering (DIANA-like)")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()