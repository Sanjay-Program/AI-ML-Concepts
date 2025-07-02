import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic normal data
X_normal, _ = make_blobs(n_samples=200, centers=1, cluster_std=0.60, random_state=42)

# Step 2: Add some anomalies
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack((X_normal, X_outliers))  # Combine normal and outlier points

# Step 3: Fit Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)  # contamination = expected % of outliers
model.fit(X)
y_pred = model.predict(X)  # Returns 1 for normal, -1 for anomaly

# Step 4: Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], c='green', label='Normal', edgecolors='k')
plt.scatter(X[y_pred == -1][:, 0], X[y_pred == -1][:, 1], c='red', label='Anomaly', edgecolors='k')
plt.title("Anomaly Detection using Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
