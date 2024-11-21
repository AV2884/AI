import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras.datasets import mnist
import numpy as np

# # Load the MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# # Choose a random image from the dataset
# random_index = np.random.randint(0, x_train.shape[0])
# selected_image = x_train[random_index]
# selected_label = y_train[random_index]

# # Plot the digit
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(selected_image, cmap='gray')

# # Remove axes ticks
# ax.set_xticks([])
# ax.set_yticks([])

# # Add grid lines and pixel values
# for i in range(selected_image.shape[0]):
#     for j in range(selected_image.shape[1]):
#         value = selected_image[i, j]
#         # Determine text and grid line color based on pixel intensity
#         if value < 128:
#             # Pixel is dark
#             text_color = 'white'
#             edge_color = 'white'
#         else:
#             # Pixel is light
#             text_color = 'black'
#             edge_color = 'black'
#         # Add a rectangle patch to simulate grid lines with the desired color
#         rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
#                          linewidth=0.5, edgecolor=edge_color, facecolor='none')
#         ax.add_patch(rect)
#         # Add pixel value text
#         ax.text(j, i, str(value), ha='center', va='center',
#                 color=text_color, fontsize=6)

# # Save and display the image
# image_with_grid_path = "mnist_digit_with_custom_grid.png"
# plt.savefig(image_with_grid_path, bbox_inches='tight', dpi=300)
# plt.show()

# print(f"Image with grid and pixel values saved as: {image_with_grid_path}")

(x_train, y_train), (_, _) = mnist.load_data()
filter_train = (y_train == 0) | (y_train == 1)
x_train_filtered = x_train[filter_train] / 255.0  # Normalize to [0, 1]
y_train_filtered = y_train[filter_train].reshape(-1, 1)
x_train_filtered = x_train_filtered.reshape(x_train_filtered.shape[0], -1)  # Flatten images
    
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
    image_path_all = '/root/aiRoot/0-AI/AI/'
    plt.savefig(image_path_all, bbox_inches='tight', dpi=80)  # Lower DPI
    plt.close()
    print(f"Saved grid image with labels")

print_sample_data()