import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

axes[0, 0].hist(titanic['age'].dropna(), bins=20, edgecolor='black', color='skyblue')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Binned Age Distribution')

axes[0, 1].hist(titanic['fare'].dropna(), bins=20, edgecolor='black', color='salmon')
axes[0, 1].set_xlabel('Fare')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Binned Fare Distribution')

sns.scatterplot(x='age', y='fare', data=titanic, ax=axes[1, 0], alpha=0.6, color='green')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Fare')
axes[1, 0].set_title('Age vs. Fare Scatter Plot')

sns.countplot(x='class', data=titanic, ax=axes[1, 1], hue='class', palette='Set3', legend=False)
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Categorical Binning - Class Bar Plot')

plt.tight_layout()
plt.show()