import numpy as np

# Step 1: Define the Activation Function (Step Function)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Step 2: Define the Perceptron Class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = learning_rate
        self.epochs = epochs

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Insert bias term
        return step_function(np.dot(self.weights, x))

    def train(self, X, y):
        for epoch in range(self.epochs):
            for inputs, label in zip(X, y):
                inputs_with_bias = np.insert(inputs, 0, 1)
                prediction = self.predict(inputs)
                self.weights += self.lr * (label - prediction) * inputs_with_bias
            # Optional: Print weights every epoch
            # print(f"Epoch {epoch+1}: Weights = {self.weights}")

# Step 3: Example Dataset (AND Gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND logic output

# Step 4: Train and Test the Perceptron
model = Perceptron(input_size=2)
model.train(X, y)

# Step 5: Predictions
print("Predictions:")
for x in X:
    print(f"{x} => {model.predict(x)}")
