import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Step 1: Generate sample data
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Step 2: Scale the data (DBSCAN is distance-based, so scaling is important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply DBSCAN
# eps: max distance for neighbors
# min_samples: minimum number of points to form a dense region
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Step 4: Visualize the clusters
# -1 label means noise
plt.figure(figsize=(8, 5))
unique_labels = set(labels)

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for label, color in zip(unique_labels, colors):
    class_member_mask = (labels == label)
    xy = X_scaled[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color),
             markeredgecolor='k', markersize=8, label=f'Cluster {label}' if label != -1 else 'Noise')

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
