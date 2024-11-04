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

'''simulation of a forward pass in neural network'''

def forward_pass(x):
    print(f"Input image vector :{x}")
    x = x.flatten().reshape(1, -1)  # Flatten and reshape x to be (1, 784)
    print(f"Input image vector flattened :{x}")

    z1 = np.dot(x,W1) + b1
    print(f"z1 = {x} * {W1} + {b1}")
    print(f"1st z1:{z1}")
    a1 = sigmoid(z1)
    print(f"1st a1 (sigmoid) :{a1}")

    z2 = np.dot(a1,W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2,W3) + b3
    a3 = sigmoid(z3)

    return a3

sample_image = x_train_filtered[1]  # Choose the first image (shape: 28x28)
a1 = forward_pass(sample_image)
print(a1)



#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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




# print_sample_data(10,10)