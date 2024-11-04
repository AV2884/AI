import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

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

(x_train,y_train) , (x_set , y_set) = mnist.load_data()
filter_train = (y_train == 0) | (y_train == 1)
x_train_filtered, y_train_filtered = x_train[filter_train], y_train[filter_train]

x_train = x_train_filtered/255
y_train = y_train_filtered

'''simulation of a forward pass in neural network'''

def forward_pass(x):
    z1 = np.dot(W1,x) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(W2,a1) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(W3,a2) + b3
    a3 = sigmoid(z3)

    return a3

sample_image = x_train_filtered[0]  # Choose the first image (shape: 28x28)
sample_image = sample_image.flatten().reshape(-1, 1)  # Flatten and reshape to (1, 784)
# Perform forward pass
a3 = forward_pass(sample_image)
# print(a3)


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




print_sample_data(50,50)