from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
data = iris.data
true_labels = iris.target

num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data)

predicted_labels = kmeans.labels_

silhouette = silhouette_score(data, predicted_labels)
rand_index = adjusted_rand_score(true_labels, predicted_labels)
davies_bouldin = davies_bouldin_score(data, predicted_labels)
wcss = kmeans.inertia_

print("Silhouette Score:", silhouette)
print("Adjusted Rand Index (ARI):", rand_index)
print("Davies-Bouldin Index:", davies_bouldin)
print("Within-Cluster Sum of Squares (WCSS):", wcss)

wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()