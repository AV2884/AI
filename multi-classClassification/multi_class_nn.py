import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import math

'''Config'''
#Data
(x_train, y_train) , (x_test , y_test) = mnist.load_data()
#Normalize and flatten the images
x_train = x_train / 255.0
x_train = x_train.reshape(x_train.shape[0],-1)
#Define NN structure
input_units = x_train.shape[1]
hidden_units_1 = 25
hidden_units_2 = 15
output_units = 10
#Count_images
#initialize weight and biases(Xaveir method)
b1 = np.zeros((1, hidden_units_1))  # (1, 25)
b2 = np.zeros((1, hidden_units_2))  # (1, 15)
b3 = np.zeros((1, output_units))    # (1, 10)
W1 = np.random.randn(input_units, hidden_units_1) * np.sqrt(1 / input_units)
W2 = np.random.randn(hidden_units_1, hidden_units_2) * np.sqrt(1 / hidden_units_1)
W3 = np.random.randn(hidden_units_2, output_units) * np.sqrt(1 / hidden_units_2)
print("Initialization complete.")

#Activation functions
def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    z -= np.max(z) # Subtract the max for numerical stability (helps prevent overflow)
    exp_z = np.exp(z)
    a_j = exp_z / np.sum(exp_z)
    return a_j


#Simulate a forward pass throught the NN
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X,W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1,W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2,W3) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, a3


#Loss function
def compute_cost(y_true, y_pred):
    m = y_train.shape[0]
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid log(0) errors
    cost = -np.sum(y_true * np.log(y_pred)) / m
    return cost

#Calculating the gradients
def compute_gradients(X, y, W1, W2, W3, z1, z2, a1, a2, a3):
    m = X.shape[0]

    #Gradient for the Output Layer
    error3 = a3 - y                                        # Shape: (m, 10) y is hot encoded
    dW3 = (1 / m) * np.dot(a2.T, error3)                   # Shape: (15, 10)
    db3 = (1 / m) * np.sum(error3, axis=0, keepdims=True)  # Shape: (1, 10)

    #Gradients for Hidden Layer 2
    error2 = np.dot(error3, W3.T) * relu_derivative(z2)
    dW2 = (1 / m) * np.dot(a1.T, error2)                   # Shape: (25, 15)
    db2 = (1 / m) * np.sum(error2, axis=0, keepdims=True)  # Shape: (1, 15)

    #Gradients for Hidden Layer 1
    error1 = np.dot(error2, W2.T) * relu_derivative(z1)    # Shape: (m, 25)
    dW1 = (1 / m) * np.dot(X.T, error1)                    # Shape: (784, 25)
    db1 = (1 / m) * np.sum(error1, axis=0, keepdims=True)  # Shape: (1, 25)

    return dW1, db1, dW2, db2, dW3, db3


#Perfrom gradient descent to minimize cost
def gradient_descent(X, y, W1, b1, W2, b2, W3, b3, learning_rate=0.001, epochs=1000):
    for i in range(epochs):
        prevoius_cost = None

        z1, a1, z2, a2, a3 = forward_pass(X, W1, b1, W2, b2, W3, b3)

        cost = compute_cost(y, a3)

        dW1, db1, dW2, db2, dW3, db3 = compute_gradients(X, y, W1, W2, W3, z1, z2, a1, a2, a3)

        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        W3 -= learning_rate * dW3
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        b3 -= learning_rate * db3

    if i % 100 == 0:
        if prevoius_cost is not None:
            print(f"Iter <{i}> : cost {cost:.5f} : Δcost {prevoius_cost - cost}")
        else:
            print(f"Iter <{i}> : cost {cost:.5f} : Δcost N/A")
        prevoius_cost = cost

    return W1, b1, W2, b2, W3, b3

def one_hot_encode(y, num_classes=10):
    '''
    0 --> [1,0,0,0,0,0,0,0,0,0]
    1 --> [0,1,0,0,0,0,0,0,0,0]
    2 --> [0,0,1,0,0,0,0,0,0,0]
    3 --> [0,0,0,1,0,0,0,0,0,0]
    '''
    y = np.array(y)  # Ensure y is a NumPy array
    one_hot = np.zeros((len(y), num_classes))
    for i, value in enumerate(y):
        one_hot[i, value] = 1.0
    return one_hot


















#-=-=-=-=-=-=--MISC-FUNC--=-=-=-=-=
def sample_data_image(rows=10, cols=10, image_path="/root/aiRoot/0-AI/AI/multi-classClassification/fig.png"):
    fig, axes = plt.subplots(rows,cols,figsize=(10,10))

    for i in range(rows * cols):
        if i % 10 == 0:
            print(f"Processing image {i}")
        image = x_train[i].reshape(28,28)
        label = y_train[i]

        axes[i // cols, i % cols].imshow(image, cmap='gray')
        axes[i // cols, i % cols].axis('off')
        axes[i // cols, i % cols].set_title(str(label), fontsize=12) 

    plt.savefig(image_path,bbox_inches='tight', dpi=100)
    plt.close()
    print("save images")

