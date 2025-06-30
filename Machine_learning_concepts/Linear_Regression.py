from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data generation
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 5 + 1.5 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = model.score(X_test, y_test)
print(f"Model Coefficients: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"R² score: {r2:.2f}")

# Visualization
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, color='red', label="Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with scikit-learn")
plt.legend()
plt.grid()
plt.show()
