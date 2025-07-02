import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
iris = load_iris()
X = iris.data   # shape (150, 4)
y = iris.target # 3 classes: 0, 1, 2

# Step 2: Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

# Step 4: Print explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

# Step 5: Visualize the reduced features
plt.figure(figsize=(8, 6))
for label, color, name in zip([0, 1, 2], ['r', 'g', 'b'], iris.target_names):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                c=color, label=name, edgecolor='k', s=60)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset (2D Projection)')
plt.legend()
plt.grid(True)
plt.show()
