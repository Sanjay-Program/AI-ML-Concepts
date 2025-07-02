import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Step 1: Load and preprocess the dataset
iris = load_iris()
X = iris.data       # 4D feature vectors
y = iris.target     # 3 classes
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Step 3: Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# Step 4: Visualization Function
def plot_2d_projection(X_proj, title, subplot_pos):
    plt.subplot(1, 2, subplot_pos)
    for label, color, name in zip([0, 1, 2], ['r', 'g', 'b'], target_names):
        plt.scatter(X_proj[y == label, 0], X_proj[y == label, 1],
                    c=color, label=name, edgecolor='k', s=60)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)

# Step 5: Plot both side-by-side
plt.figure(figsize=(14, 6))
plot_2d_projection(X_tsne, "t-SNE Visualization", 1)
plot_2d_projection(X_umap, "UMAP Visualization", 2)
plt.tight_layout()
plt.show()
