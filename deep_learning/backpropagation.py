import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Set random seed for reproducibility
np.random.seed(42)

# Network architecture
input_size = 2
hidden_size = 2
output_size = 1
lr = 0.1
epochs = 10000

# Initialize weights and biases
W1 = np.random.uniform(size=(input_size, hidden_size))   # (2x2)
b1 = np.zeros((1, hidden_size))                          # (1x2)
W2 = np.random.uniform(size=(hidden_size, output_size))  # (2x1)
b2 = np.zeros((1, output_size))                          # (1x1)

# Training loop
for epoch in range(epochs):
    # Forward Pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # output

    # Compute error
    error = y - a2

    # Backward Pass
    d_a2 = error * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_a2)
    d_b2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_a1)
    d_b1 = np.sum(d_a1, axis=0)

    # Update weights and biases
    W2 += lr * d_W2
    b2 += lr * d_b2
    W1 += lr * d_W1
    b1 += lr * d_b1

    # Optional: print loss occasionally
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions
print("\nPredictions after training:")
print(a2.round())
