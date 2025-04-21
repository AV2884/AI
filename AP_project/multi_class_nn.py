import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os
import time
import os

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)

# Define NN architecture
input_units     = x_train.shape[1]
hidden_units_1  = 25
hidden_units_2  = 15
output_units    = 10

# Helper functions
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"

def one_hot_encode(y, num_classes=10):
    y = np.array(y)
    one_hot = np.zeros((len(y), num_classes))
    for i, value in enumerate(y):
        one_hot[i, value] = 1.0
    return one_hot

def load_or_initialize_weights():
    if all(os.path.exists(file) for file in weights_and_biases):
        print("Loaded saved weights and biases.")
        W1 = np.load("models/W1.npy"); b1 = np.load("models/b1.npy")
        W2 = np.load("models/W2.npy"); b2 = np.load("models/b2.npy")
        W3 = np.load("models/W3.npy"); b3 = np.load("models/b3.npy")
    else:
        print("Initializing new weights and biases.")
        W1 = np.random.randn(input_units, hidden_units_1) * np.sqrt(1 / input_units)
        W2 = np.random.randn(hidden_units_1, hidden_units_2) * np.sqrt(1 / hidden_units_1)
        W3 = np.random.randn(hidden_units_2, output_units) * np.sqrt(1 / hidden_units_2)
        b1 = np.zeros((1, hidden_units_1))
        b2 = np.zeros((1, hidden_units_2))
        b3 = np.zeros((1, output_units))
    return W1, b1, W2, b2, W3, b3

def calculate_accuracy(X, y, W1, b1, W2, b2, W3, b3):
    _, _, _, _, a3 = forward_pass(X, W1, b1, W2, b2, W3, b3)
    predictions = np.argmax(a3, axis=1)
    true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == true_labels) * 100
    return accuracy


# Activation functions
def relu(z): 
    return np.maximum(0, z)
def relu_derivative(z): 
    return (z > 0).astype(float)
def softmax(z, temperature=1.5):
    z = z / temperature
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Forward pass
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, a3


# Cost function
def compute_cost(y_true, y_pred):
    m = y_train.shape[0]
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    cost = -np.sum(y_true * np.log(y_pred)) / m
    return cost


# Backpropagation
def compute_gradients(X, y, W1, W2, W3, z1, z2, a1, a2, a3):
    m = X.shape[0]
    error3 = a3 - y
    dW3 = (1 / m) * np.dot(a2.T, error3)
    db3 = (1 / m) * np.sum(error3, axis=0, keepdims=True)

    error2 = np.dot(error3, W3.T) * relu_derivative(z2)
    dW2 = (1 / m) * np.dot(a1.T, error2)
    db2 = (1 / m) * np.sum(error2, axis=0, keepdims=True)

    error1 = np.dot(error2, W2.T) * relu_derivative(z1)
    dW1 = (1 / m) * np.dot(X.T, error1)
    db1 = (1 / m) * np.sum(error1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


# Training loop
def gradient_descent(X, y, W1, b1, W2, b2, W3, b3, learning_rate=0.01, epochs=1000):
    previous_cost = None
    start_time = time.time()
    costs, iterations = [], []

    for i in range(epochs):
        z1, a1, z2, a2, a3 = forward_pass(X, W1, b1, W2, b2, W3, b3)
        cost = compute_cost(y, a3)
        if i % 100 == 0:
            costs.append(cost)
            iterations.append(i)
        dW1, db1, dW2, db2, dW3, db3 = compute_gradients(X, y, W1, W2, W3, z1, z2, a1, a2, a3)
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        W3 -= learning_rate * dW3
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        b3 -= learning_rate * db3
        total_time = time.time() - start_time
        avg_time_per_iter = total_time / (i + 1)
        etc = (epochs - i - 1) * avg_time_per_iter
        if i % 10 == 0:
            delta = previous_cost - cost if previous_cost else 0
            print(f"Iter <{i:5d}> : cost {cost:.7f} : Δcost {delta:.7f} | ETC: {format_time(etc):>8} | T: {format_time(total_time):>8}")
            previous_cost = cost
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds.")
    os.makedirs("models", exist_ok=True)
    np.save("models/W1.npy", W1)
    np.save("models/b1.npy", b1)
    np.save("models/W2.npy", W2)
    np.save("models/b2.npy", b2)
    np.save("models/W3.npy", W3)
    np.save("models/b3.npy", b3)
    print("Final weights and biases saved.")
    plt.plot(iterations, costs, marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.grid(True)
    plt.savefig("cost_vs_iterations.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("Cost graph saved as 'cost_vs_iterations.png'.")
    return W1, b1, W2, b2, W3, b3, training_time

# Main Execution
# ---- Parameters ----
mode = "t"    # t -> training p -> for prediction
learning_rate = 0.00889
num_epochs    = 1_00_000
# ---------------------

weights_and_biases = ["models/W1.npy", "models/b1.npy", "models/W2.npy", "models/b2.npy", "models/W3.npy", "models/b3.npy"]
y_train_one_hot = one_hot_encode(y_train)

if mode == "t":
    clear_terminal()
    W1, b1, W2, b2, W3, b3 = load_or_initialize_weights()

    initial_cost = compute_cost(y_train_one_hot, forward_pass(x_train, W1, b1, W2, b2, W3, b3)[-1])
    W1, b1, W2, b2, W3, b3, training_time = gradient_descent(
        x_train, y_train_one_hot, W1, b1, W2, b2, W3, b3, learning_rate, num_epochs
    )
    final_cost = compute_cost(y_train_one_hot, forward_pass(x_train, W1, b1, W2, b2, W3, b3)[-1])

    print(f"\nInitial Cost:  {initial_cost:.6f}")
    print(f"Final Cost:    {final_cost:.6f}")
    print(f"Training Time: {format_time(training_time)}")

    accuracy = calculate_accuracy(x_train, y_train_one_hot, W1, b1, W2, b2, W3, b3)
    print(f"Training Accuracy: {accuracy:.2f}%")

elif mode == "p":
    W1 = np.load("models/W1.npy"); b1 = np.load("models/b1.npy")
    W2 = np.load("models/W2.npy"); b2 = np.load("models/b2.npy")
    W3 = np.load("models/W3.npy"); b3 = np.load("models/b3.npy")

    print("Weights and biases loaded for prediction.")
    user_input = input("Enter the range of indices to predict (e.g. 0-9): ")
    start, end = map(int, user_input.split('-'))

    sample_images = x_test[start:end+1]
    sample_labels = y_test[start:end+1]
    _, _, _, _, predictions = forward_pass(sample_images.reshape(sample_images.shape[0], -1), W1, b1, W2, b2, W3, b3)
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(len(predicted_labels)):
        print(f"\nImage {start + i}:")
        print(f"True Label: {sample_labels[i]} | Predicted: {predicted_labels[i]}")
        print("Softmax Probabilities:")
        for digit, prob in enumerate(predictions[i]):
            tag = "<-- Predicted" if digit == predicted_labels[i] else ""
            print(f"  {digit}: {prob * 100:.2f}% {tag}")

    # Optional: Show images
    fig, axes = plt.subplots(1, len(sample_images), figsize=(3 * len(sample_images), 3))
    for i in range(len(sample_images)):
        axes[i].imshow(sample_images[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Pred: {predicted_labels[i]}\nTrue: {sample_labels[i]}")
    plt.tight_layout()
    plt.savefig("prediction_output.png", bbox_inches='tight')
    plt.close()
