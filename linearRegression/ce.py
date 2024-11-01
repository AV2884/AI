import numpy as np
from pprint import pprint

# Generating synthetic sample data
np.random.seed(42)  # For reproducibility

num_features = 4
num_samples = 50

# Input features: [Underlying Price, OI, Epoch Time, IV]
x_train_options = np.column_stack((
    np.random.uniform(19000, 20000, num_samples),  # Nifty Spot Price (S)
    np.random.randint(1000, 50000, num_samples),  # Open Interest (OI)
    np.random.randint(1_695_000_000, 1_700_000_000, num_samples),  # Time (Epoch)
    np.random.uniform(0.1, 0.3, num_samples)  # Implied Volatility (IV)
))

# Option premiums (Y) - generated with some noise
y_train_options = (
    0.01 * x_train_options[:, 0] + 0.001 * x_train_options[:, 1] +
    0.5 * x_train_options[:, 3] + np.random.normal(0, 10, num_samples)
)

# Normalize the input features and target
def normalize(data):
    mean = np.mean(data, axis=0)
    range_ = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - mean) / range_

x_normalized = normalize(x_train_options)
mean_y = np.mean(y_train_options)
range_y = np.max(y_train_options) - np.min(y_train_options)
y_normalized = (y_train_options - mean_y) / range_y

# Model function
def F(w, b, x):
    return np.dot(w, x) + b

# Cost function
def cost_function(w, b):
    cost = 0
    for i in range(num_samples):
        predicted = F(w, b, x_normalized[i])
        error = predicted - y_normalized[i]
        cost += error ** 2
    return cost / (2 * num_samples)

# Gradient Descent for training
def gradient_descent(w, b, alpha, num_iterations):
    for epoch in range(num_iterations):
        dw = np.zeros_like(w)
        db = 0

        for i in range(num_samples):
            prediction = F(w, b, x_normalized[i])
            error = prediction - y_normalized[i]

            for j in range(num_features):
                dw[j] += error * x_normalized[i, j]
            db += error

        w -= alpha * dw / num_samples
        b -= alpha * db / num_samples

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Cost: {cost_function(w, b)}")

    return w, b

# Initialize parameters
w = np.zeros(num_features)
b = 0
alpha = 0.01
num_iterations = 10_000

# Train the model
w, b = gradient_descent(w, b, alpha, num_iterations)

# Prediction with unnormalization
def predict_unnormalized(x, w, b):
    normalized_pred = F(w, b, x)
    return normalized_pred * range_y + mean_y

# Test the model with user input
while True:
    print("\nNifty 50 Option Price Prediction")
    S = float(input("Enter underlying asset price (S): "))
    OI = int(input("Enter open interest (OI): "))
    epoch_time = int(input("Enter time in epoch format: "))
    iv = float(input("Enter implied volatility (IV): "))

    # Normalize the input
    usecase = np.array([S, OI, epoch_time, iv], dtype=float)
    usecase_normalized = normalize(usecase)

    # Predict the option premium
    predicted_premium = predict_unnormalized(usecase_normalized, w, b)
    print(f"Predicted Option Premium: ${predicted_premium:.2f}")
