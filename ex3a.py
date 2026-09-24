import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

iris = load_iris()
data = iris.data

num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data)

kmeans_labels = kmeans.labels_
kmeans_centroids = kmeans.cluster_centers_

dist_matrix = np.linalg.norm(
    data[:, np.newaxis] - data,
    axis=-1
)

mst = minimum_spanning_tree(csr_matrix(dist_matrix))

connectivity_matrix = mst + mst.T
connectivity_matrix = connectivity_matrix.toarray()

agg_clustering = AgglomerativeClustering(
    n_clusters=num_clusters,
    connectivity=connectivity_matrix
)

mst_labels = agg_clustering.fit_predict(data)

print("K-Means Cluster Labels:", kmeans_labels)
print("K-Means Cluster Centroids:\n", kmeans_centroids)
print("MST-based Agglomerative Cluster Labels:", mst_labels)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("K-Means Clustering")
plt.scatter(
    reduced_data[:, 0],
    reduced_data[:, 1],
    c=kmeans_labels,
    cmap='viridis'
)

plt.subplot(1, 2, 2)
plt.title("MST-based Agglomerative Clustering")
plt.scatter(
    reduced_data[:, 0],
    reduced_data[:, 1],
    c=mst_labels,
    cmap='plasma'
)

plt.tight_layout()
plt.show()