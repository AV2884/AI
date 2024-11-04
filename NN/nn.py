import numpy as np
from tensorflow.keras.datasets import mnist
import time

'''Data'''
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
    start_time = time.time()
    for i in range(num_epochs):
        
        # Forward pass
        a1, a2, a3 = forward_pass(x_train, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")

        # Compute cost
        cost = compute_cost(y_train, a3)

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

        if i % 100 == 0:
            print(f"Iter: {i}, Cost: {cost}")

    end_time = time.time()
    training_time = end_time - start_time
    return W1, b1, W2, b2, W3, b3, training_time

    

def print_sample_data(rows=10,cols=10):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    status = 0
    for i in range(rows * cols):
        status += 1
        if i % 100 == 0:
            print(status)
        axes[i // cols, i % cols].imshow(x_train_filtered[i], cmap='gray')
        axes[i // cols, i % cols].axis('off')

    # Save the image of the grid of 100 images of 0 and 1
    image_path_all = '/root/aiRoot/0-AI/AI/NN/mnist_all_0_and_1_samples.png'
    plt.savefig(image_path_all, bbox_inches='tight')
    plt.close()
    print(f"Saved grid image")


def model_summary(W1, b1, W2, b2, W3, b3, training_time):
    print("\nModel Summary:")
    print(f"Number of Training Images: {num_images}")
    print(f"Number of '0' Images: {num_zeros}")
    print(f"Number of '1' Images: {num_ones}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Weight and Bias Parameters Used:")
    print(f"  - W1: {W1.shape}, b1: {b1.shape}")
    print(f"  - W2: {W2.shape}, b2: {b2.shape}")
    print(f"  - W3: {W3.shape}, b3: {b3.shape}")
    print()


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

init_method = "x"       # "x" for xavier or "r" random
activation = "s"        # "s" for sigmoid or "r" for relu
mode = "t"              # "t" for train or "p" for predict
a = 0.003
epoch = 100

if init_method == "x":
    W1 = np.random.randn(input_units, hidden_units_1) * np.sqrt(1 / input_units)
    W2 = np.random.randn(hidden_units_1, hidden_units_2) * np.sqrt(1 / hidden_units_1)
    W3 = np.random.randn(hidden_units_2, output_units) * np.sqrt(1 / hidden_units_2)
else:  # Random initialization
    W1 = np.random.randn(input_units, hidden_units_1) * 0.01
    W2 = np.random.randn(hidden_units_1, hidden_units_2) * 0.01
    W3 = np.random.randn(hidden_units_2, output_units) * 0.01
b1 = np.zeros((1, hidden_units_1))  # Shape: (1, 25)
b2 = np.zeros((1, hidden_units_2))  # Shape: (1, 15)
b3 = np.zeros((1, output_units)) 

activation_hidden = "sigmoid" if activation == "s" else "relu"
if mode == "t":
        W1, b1, W2, b2, W3, b3, training_time = gradient_descent(
            x_train_filtered, y_train_filtered, W1, b1, W2, b2, W3, b3, activation_hidden,learning_rate=a,num_epochs=epoch
        )
        model_summary(W1, b1, W2, b2, W3, b3, training_time)
elif mode == "p":
    # For prediction, you can implement a function that uses the trained weights
    print("Prediction mode selected. Implement prediction logic here.")
else:
    print("Invalid mode selected. Please choose 'train' or 'predict'.")




# print_sample_data(10,10)