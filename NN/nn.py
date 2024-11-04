import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

'''Data'''
(x_train,y_train) , (x_set , y_set) = mnist.load_data()
filter_train = (y_train == 0) | (y_train == 1)
x_train_filtered, y_train_filtered = x_train[filter_train], y_train[filter_train]
#Normalize 
x_train_filtered = x_train_filtered/255 

''' Avoid Large Initial Values: If we start with large weights, 
    the outputs of each layer could explode. '''
W1 = np.random.randn(784,25) * 0.01
b1 = np.zeros(25)

W2 = np.random.randn(25,15) * 0.01
b2 = np.zeros(15)

W3 = np.random.randn(15,1) * 0.01
b3 = np.zeros(1)

#Activation
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))


#cost function
def compute_cost(y , y_hat):
    m = y.shape[0]
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10) #limits the values in an array to a specified range
    '''Non vectorized approach'''
    # cost = 0
    # for i in range(m):
    #     loss = (y[i] * np.log(y_hat[i])) + ((1 - y[i])*np.log(1-y_hat[i]))
    #     cost += (-1 / m) * loss
    '''Vectorized'''
    cost = (-1 / m) * np.sum( (y * np.log(y_hat))  + (1 - y) *  np.log(1 - y_hat))
    return cost 


#compute_gradient
def compute_gradient(x, y, W1, b1, W2, b2, W3, b3):
    # Forward pass to get activations
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    # Compute the error at the output layer
    error = a3 - y
    # print("1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",error.shape)

    # Number of training examples (m)
    m = y.shape[0]

    # Gradients for W3 and b3
    dW3 = (1 / m) * np.dot(a2.T, error)
    db3 = (1 / m) * np.sum(error, axis=0)

    # Backpropagate the error to the second hidden layer
    dz2 = np.dot(error, W3.T) * (a2 * (1 - a2))
    dW2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0)

    # Backpropagate the error to the first hidden layer
    dz1 = np.dot(dz2, W2.T) * (a1 * (1 - a1))
    dW1 = (1 / m) * np.dot(x.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0)

    # Return all gradients
    return dW1, db1, dW2, db2, dW3, db3


#gradient_descent
def gradient_descent(x_train,y_train,W1, b1, W2, b2, W3, b3):
    learning_rate = 0.001
    num_epochs = 1000
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = y_train.reshape(-1, 1)

    for i in range(num_epochs):
        a3 = forward_pass(x_train)
        cost = compute_cost(y_train , a3)
        dW1, db1, dW2, db2, dW3, db3 = compute_gradient(x_train, y_train, W1, b1, W2, b2, W3, b3)

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W3 = W3 - learning_rate * dW3
        b3 = b3 - learning_rate * db3

        if i % 100 == 0:
            print(f"Iter: {i} , Cost: {cost}")

    return W1, b1, W2, b2, W3, b3

#Forward pass through the NN
def forward_pass(x):
    z1 = np.dot(x, W1) + b1  # (m, 784) dot (784, 25) -> (m, 25)
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2  # (m, 25) dot (25, 15) -> (m, 15)
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3  # (m, 15) dot (15, 1) -> (m, 1)
    a3 = sigmoid(z3)

    return a3
    

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


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Make sure x_train_filtered is flattened and normalized
x_train_filtered = x_train_filtered.reshape(x_train_filtered.shape[0], -1)  # Flatten images
y_train_filtered = y_train_filtered.reshape(-1, 1) 

# Call the gradient_descent function
W1, b1, W2, b2, W3, b3 = gradient_descent(x_train_filtered, y_train_filtered, W1, b1, W2, b2, W3, b3)

# print_sample_data(10,10)