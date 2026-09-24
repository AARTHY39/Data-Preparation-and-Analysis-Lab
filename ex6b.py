import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def PCA(X, num_components):
    X_meaned = X - np.mean(X, axis=0)

    cov_mat = np.cov(X_meaned, rowvar=False)

    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    X_reduced = np.dot(X_meaned, eigenvector_subset)

    return X_reduced

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

data = pd.read_csv(
    url,
    names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
)

x = data.iloc[:, 0:4].values
target = data.iloc[:, 4]

mat_reduced = PCA(x, 2)

principal_df = pd.DataFrame(mat_reduced, columns=['PC1', 'PC2'])

principal_df = pd.concat(
    [principal_df, pd.DataFrame(target, columns=['target'])],
    axis=1
)

plt.figure(figsize=(6, 6))

sb.scatterplot(
    data=principal_df,
    x='PC1',
    y='PC2',
    hue='target',
    s=60,
    palette='icefire'
)

plt.title("PCA of Iris Dataset")
plt.show()