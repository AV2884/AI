import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time
import os
import sys

#Load data
(x_train, y_train), (x_set, y_set) = mnist.load_data()
filter_train = (y_train == 0) | (y_train == 1)
x_train_filtered, y_train_filtered = x_train[filter_train], y_train[filter_train]

# Normalize the images and flatten them
x_train_filtered = x_train_filtered / 255.0
x_train_filtered = x_train_filtered.reshape(x_train_filtered.shape[0], -1)  # Flatten images
y_train_filtered = y_train_filtered.reshape(-1, 1)

# Define the number of units per layer
input_units = x_train_filtered.shape[1]  # 784
hidden_units_1 = 25
hidden_units_2 = 15
output_units = 1

# Count the number of images for each class
num_images = x_train_filtered.shape[0]
num_zeros = np.sum(y_train_filtered == 0)
num_ones = np.sum(y_train_filtered == 1)

# Activation functions choice
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def relu(z):
    return np.maximum(0, z)
def activation_function(z, activation_type):
    if activation_type == "relu":
        return relu(z)
    elif activation_type == "sigmoid":
        return sigmoid(z)
    else:
        raise ValueError("Invalid activation type. Use 'relu' or 'sigmoid'.")

#cost function
def compute_cost(y , y_hat):
    m = y.shape[0]
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10) #limits the values in an array to a specified range
    '''Vectorized'''
    cost = (-1 / m) * np.sum( (y * np.log(y_hat))  + (1 - y) *  np.log(1 - y_hat))
    return cost 


#Forward pass through the NN
def forward_pass(x, W1, b1, W2, b2, W3, b3, activation_hidden, activation_output):
    z1 = np.dot(x, W1) + b1
    a1 = activation_function(z1, activation_hidden)
    z2 = np.dot(a1, W2) + b2
    a2 = activation_function(z2, activation_hidden)
    z3 = np.dot(a2, W3) + b3
    a3 = activation_function(z3, activation_output)
    return a1, a2, a3


#Computing the gradient 
def compute_gradient(x, y, a1, a2, a3, W3, W2, activation_hidden):
    m = y.shape[0]
    error = a3 - y

    # Gradients for W3 and b3
    dW3 = (1 / m) * np.dot(a2.T, error)
    db3 = (1 / m) * np.sum(error, axis=0, keepdims=True)

    # Backpropagate to the second hidden layer
    if activation_hidden == "relu":
        dz2 = np.dot(error, W3.T) * (a2 > 0).astype(float)
    else:  # Sigmoid
        dz2 = np.dot(error, W3.T) * (a2 * (1 - a2))
    dW2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

    # Backpropagate to the first hidden layer
    if activation_hidden == "relu":
        dz1 = np.dot(dz2, W2.T) * (a1 > 0).astype(float)
    else:  # Sigmoid
        dz1 = np.dot(dz2, W2.T) * (a1 * (1 - a1))
    dW1 = (1 / m) * np.dot(x.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


# Gradient descent function
def gradient_descent(x_train, y_train, W1, b1, W2, b2, W3, b3, activation_hidden,learning_rate=0.01,num_epochs=1000):
    costs = []
    start_time = time.time()
    previous_cost = None
    for i in range(num_epochs):
        
        # Forward pass
        a1, a2, a3 = forward_pass(x_train, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")

        # Compute cost
        cost = compute_cost(y_train, a3)
        costs.append(cost)

        # Compute gradients
        dW1, db1, dW2, db2, dW3, db3 = compute_gradient(x_train, y_train, a1, a2, a3, W3, W2, activation_hidden)

        # Update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        end_time = time.time()
        elapsed_time = end_time - start_time

        if previous_cost is not None:
            delta_cost = previous_cost - cost
            # delta_percentage = (delta_cost / previous_cost) * 100
            print(f"Iter: {i}, Cost: {cost:.7f}, D:{delta_cost:.10f}, t:{elapsed_time:.2f}s")
        else:
            print(f"Iter: {i}, Cost: {cost:.7f}, Time: {elapsed_time:.2f}s")
        previous_cost = cost
        # if i % 10 == 0:
        #     print(f"Iter: {i}, Cost: {cost}")

    end_time = time.time()
    training_time = end_time - start_time
    return W1, b1, W2, b2, W3, b3, training_time, costs

    
def print_sample_data(rows=10, cols=10):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    
    for i in range(rows * cols):
        if i % 10 == 0:
            print(f"Processing image {i}")
        # Reshape the flattened image back to (28, 28)
        image = x_train_filtered[i].reshape(28, 28)
        label = y_train_filtered[i][0]  # Get the corresponding label
        
        axes[i // cols, i % cols].imshow(image, cmap='gray')
        axes[i // cols, i % cols].axis('off')
        axes[i // cols, i % cols].set_title(str(label), fontsize=12)  # Larger font size

    # Save the image of the grid of images and labels
    image_path_all = '/root/aiRoot/0-AI/AI/NN/mnist_all_0_and_1_samples_with_labels.png'
    plt.savefig(image_path_all, bbox_inches='tight', dpi=80)  # Lower DPI
    plt.close()
    print(f"Saved grid image with labels")


def model_summary(W1, b1, W2, b2, W3, b3, training_time, initial_cost, final_cost, learning_rate, num_epochs, activation_hidden, init_method,accuracy):
    # Calculate total parameters
    total_parameters = (W1.size + b1.size) + (W2.size + b2.size) + (W3.size + b3.size)
    cost_reduction = ((initial_cost - final_cost) / initial_cost) * 100  # Percentage reduction

    memory_usage_bytes = W1.nbytes + b1.nbytes + W2.nbytes + b2.nbytes + W3.nbytes + b3.nbytes

    # Estimate memory usage for activations and gradients
    # Assuming float64 (8 bytes per value)
    activation_memory_bytes = (x_train_filtered.nbytes + 
                               W1.shape[1] * x_train_filtered.shape[0] * 8 +  # a1
                               W2.shape[1] * x_train_filtered.shape[0] * 8 +  # a2
                               output_units * x_train_filtered.shape[0] * 8)  # a3

    gradient_memory_bytes = memory_usage_bytes  # Roughly equal to weights and biases

    # Total memory usage
    total_memory_usage_bytes = memory_usage_bytes + activation_memory_bytes + gradient_memory_bytes
    total_memory_usage_kb = total_memory_usage_bytes / 1024  # Convert to kilobytes
    total_memory_usage_mb = total_memory_usage_kb / 1024  # Convert to megabytes


    print("\nModel Summary:")
    print(f"Number of Training Images: {num_images}")
    print(f"Number of '0' Images: {num_zeros}")
    print(f"Number of '1' Images: {num_ones}")
    print(f"Input Image Shape: (784,) (Flattened from 28x28)")
    print("Normalization: Pixel values scaled to [0, 1]\n")

    print("Training Configuration:")
    print(f"- Activation Function (Hidden Layers): {activation_hidden.capitalize()}")
    print(f"- Activation Function (Output Layer): Sigmoid")
    print(f"- Weight Initialization: {init_method.capitalize()}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Number of Epochs: {num_epochs}")
    print(f"- Initial Cost Value: {initial_cost:.4f}")
    print(f"- Final Cost Value: {final_cost:.4f}")
    print(f"- Cost Reduction: {cost_reduction:.2f}%")
    print(f"- Accuracy: {accuracy:.5f}%\n")
    print(f"- Training Time: {training_time:.2f} seconds\n")

    print("Weight and Bias Parameters:")
    print(f"- W1: {W1.shape}, b1: {b1.shape}")
    print(f"- W2: {W2.shape}, b2: {b2.shape}")
    print(f"- W3: {W3.shape}, b3: {b3.shape}")
    print(f"- Total Parameters: {total_parameters}\n")
    print(f"- Space Used by Weights and Biases: {memory_usage_bytes / 1024:.2f} KB")
    print(f"- Estimated Total Memory Usage During Training: {total_memory_usage_mb:.2f} MB\n")

    print("Environment Details:")
    print("- Hardware: CPU")  # Update this if you use GPU in the future
    print("- Python Version:", sys.version)
    print("- NumPy Version:", np.__version__)


def calculate_accuracy(x_train, y_train, W1, b1, W2, b2, W3, b3, activation_hidden):
    _, _, a3 = forward_pass(x_train, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")
    predictions = (a3 > 0.5).astype(int)  # Convert probabilities to binary output
    correct_predictions = np.sum(predictions == y_train)
    accuracy = (correct_predictions / y_train.shape[0]) * 100  # Calculate accuracy percentage
    return accuracy


def predict(x, W1, b1, W2, b2, W3, b3, activation_hidden):
    # Perform a forward pass with the trained weights and biases
    _, _, a3 = forward_pass(x, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")
    
    # Use a threshold of 0.5 to make binary predictions
    predictions = (a3 >= 0.5).astype(int)
    return predictions


def display_predictions(sample_images, predictions, sample_labels):
    num_samples = len(sample_images)  # Get the number of images provided

    # Create a figure to display predictions
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))  # Adjust the figure size dynamically

    if num_samples == 1:
        # Handle the case for a single image
        image = sample_images[0].reshape(28, 28)  # Reshape to 28x28 for display
        axes.imshow(image, cmap='gray')
        axes.axis('off')
        axes.set_title(f"Pred: {predictions[0][0]}, True: {sample_labels[0][0]}")
    else:
        # Handle the case for multiple images
        for i in range(num_samples):
            image = sample_images[i].reshape(28, 28)  # Reshape to 28x28 for display
            axes[i].imshow(image, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Pred: {predictions[i][0]}, True: {sample_labels[i][0]}")
            # Save the image of predictions

    image_path_predictions = '/root/aiRoot/0-AI/AI/binaryClassification/data/predictions_sample.png'
    plt.savefig(image_path_predictions, bbox_inches='tight')
    plt.close()

    print(f"Saved sample predictions image at: {image_path_predictions}")


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Configuration settings
init_method = "r"       # "x" for Xavier or "r" for random initialization
activation = "s"        # "s" for sigmoid or "r" for ReLU
mode = "t"              # "t" for train or "p" for predict
learning_rate = 0.001
num_epochs = 10_000

# Set the activation function for hidden layers
activation_hidden = "sigmoid" if activation == "s" else "relu"

# Check if saved weights and biases exist
if mode == "t":
    weights_and_biases = ["models/W1.npy", "models/b1.npy", "models/W2.npy", "models/b2.npy", "models/W3.npy", "models/b3.npy"]
    if all(os.path.exists(file) for file in weights_and_biases):
        # Load weights and biases
        W1 = np.load("models/W1.npy")
        b1 = np.load("models/b1.npy")
        W2 = np.load("models/W2.npy")
        b2 = np.load("models/b2.npy")
        W3 = np.load("models/W3.npy")
        b3 = np.load("models/b3.npy")
        print("Loaded saved weights and biases.")
    else:
        print("No saved model found")
        print("Initializing W and B")
        # Initialize weights and biases
        if init_method == "x":
            # Xavier initialization
            W1 = np.random.randn(input_units, hidden_units_1) * np.sqrt(1 / input_units)
            W2 = np.random.randn(hidden_units_1, hidden_units_2) * np.sqrt(1 / hidden_units_1)
            W3 = np.random.randn(hidden_units_2, output_units) * np.sqrt(1 / hidden_units_2)
        else:  # Random initialization
            W1 = np.random.randn(input_units, hidden_units_1) * 0.01
            W2 = np.random.randn(hidden_units_1, hidden_units_2) * 0.01
            W3 = np.random.randn(hidden_units_2, output_units) * 0.01

        b1 = np.zeros((1, hidden_units_1))  # Shape: (1, 25)
        b2 = np.zeros((1, hidden_units_2))  # Shape: (1, 15)
        b3 = np.zeros((1, output_units))    # Shape: (1, 1)

    # Train the model
    initial_cost = compute_cost(
        y_train_filtered,
        forward_pass(x_train_filtered, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")[2]
    )
    W1, b1, W2, b2, W3, b3, training_time,costs = gradient_descent(
        x_train_filtered, y_train_filtered, W1, b1, W2, b2, W3, b3, activation_hidden, learning_rate=learning_rate, num_epochs=num_epochs
    )
    final_cost = compute_cost(
        y_train_filtered,
        forward_pass(x_train_filtered, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")[2]
    )

    iterations = range(1, len(costs) + 1)
    plt.plot(iterations, costs, label="Cost")
    plt.title("Cost vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig("cost_iter-graph.png", bbox_inches='tight',dpi=800)

    # Save the trained weights and biases
    np.save("models/W1.npy", W1)
    np.save("models/b1.npy", b1)
    np.save("models/W2.npy", W2)
    np.save("models/b2.npy", b2)
    np.save("models/W3.npy", W3)
    np.save("models/b3.npy", b3)
    print("Saved weights and biases after training.")

    # Display the model summary
    accuracy = calculate_accuracy(x_train_filtered, y_train_filtered, W1, b1, W2, b2, W3, b3, activation_hidden)
    model_summary(W1, b1, W2, b2, W3, b3, training_time, initial_cost, final_cost, learning_rate, num_epochs, activation_hidden, init_method, accuracy)

if mode == "p":
    print("Prediction mode selected. Making predictions...")

    # Load saved weights and biases
    W1 = np.load("models/W1.npy")
    b1 = np.load("models/b1.npy")
    W2 = np.load("models/W2.npy")
    b2 = np.load("models/b2.npy")
    W3 = np.load("models/W3.npy")
    b3 = np.load("models/b3.npy")
    print("Loaded saved weights and biases.")

    # Ask the user for the range of images to predict
    user_input = input("Enter the range of indices (e.g., '0-4' for images 0 to 4 or '0-0' for a single image): ")
    
    # Parse the range
    start, end = map(int, user_input.split('-'))
    sample_images = x_train_filtered[start:end+1]  # Select images from start to end (inclusive)
    sample_labels = y_train_filtered[start:end+1]  # Select corresponding labels

    # Forward pass for prediction
    _, _, predictions = forward_pass(sample_images, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")
    formatted_predictions = [f"{float(pred[0]):.8f}" for pred in predictions]

    for i in range(len(formatted_predictions)):
        binary_output = 1 if float(formatted_predictions[i]) > 0.5 else 0
        print(f"PREDICTION {i} -> {formatted_predictions[i]} = {binary_output}")

    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

    # Call the function to display predictions
    display_predictions(sample_images, predictions, sample_labels)

else:
    # Load an image of the number 2 from the MNIST dataset
    W1 = np.load("models/W1.npy")
    b1 = np.load("models/b1.npy")
    W2 = np.load("models/W2.npy")
    b2 = np.load("models/b2.npy")
    W3 = np.load("models/W3.npy")
    b3 = np.load("models/b3.npy")

    # Load the MNIST test data
    (_, _), (x_test, y_test) = mnist.load_data()

    # Filter for images labeled as '8'
    number_8_images = x_test[y_test == 8]

    # Take the first "8" image for testing
    image_of_8 = number_8_images[0]

    # Preprocess the image: normalize and flatten
    image_of_8 = image_of_8 / 255.0
    image_of_8 = image_of_8.flatten().reshape(1, -1)

    # Make a prediction using the forward_pass function
    _, _, prediction = forward_pass(image_of_8, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")

    print(f"Model's output for the number '8': {prediction[0][0]:.4f}")

# Plot sample data (if needed)
# print_sample_data(50, 50)
