import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

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

