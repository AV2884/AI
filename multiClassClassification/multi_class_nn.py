import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import sys
import os
import time
from tqdm import tqdm
from PIL import Image,ImageOps

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

image_path="/root/aiRoot/0-AI/AI/multiClassClassification/plots/fig.png"

#Activation functions
def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # z -= np.max(z) # Subtract the max for numerical stability (helps prevent overflow)
    # exp_z = np.exp(z)
    # a_j = exp_z / np.sum(exp_z)
    # return a_j
    z -= np.max(z, axis=1, keepdims=True)  # Stability improvement
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


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
    # cost = -np.sum(y_true * np.log(y_pred)) / m <--- too slow
    return -np.sum(np.sum(y_true * np.log(y_pred), axis=1)) / m 


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
# def gradient_descent(X, y, W1, b1, W2, b2, W3, b3, learning_rate=0.01, epochs=1000):
    save_folder = "checkpoints"
    os.makedirs(save_folder, exist_ok=True)
    previous_cost = None
    image_path_cost = "/root/aiRoot/0-AI/AI/multi-classClassification/checkpoints"
    save_interval = 500
    start_time = time.time()
    iter_start_time = time.time()  # To track time per iteration

    costs = []

    for i in tqdm(range(epochs), desc="Training Progress", position=0, leave=True):
        z1, a1, z2, a2, a3 = forward_pass(X, W1, b1, W2, b2, W3, b3)
        cost = compute_cost(y, a3)
        costs.append(cost)
        dW1, db1, dW2, db2, dW3, db3 = compute_gradients(X, y, W1, W2, W3, z1, z2, a1, a2, a3)

        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        W3 -= learning_rate * dW3
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        b3 -= learning_rate * db3

        # Calculate elapsed time and estimated remaining time
        elapsed_time = time.time() - iter_start_time
        total_elapsed_time = time.time() - start_time
        remaining_iters = epochs - i
        estimated_completion_time = (total_elapsed_time / (i + 1)) * remaining_iters

        # Print status
        if i % 10 == 0:
            if previous_cost is not None:
                delta_cost = previous_cost - cost
                tqdm.write(
                    f"Iter <{i:5d}> : cost {cost:.7f} : Δcost {delta_cost: .7f} | "
                    f"ETC: {format_time(estimated_completion_time):>8} | T: {format_time(total_elapsed_time):>8}"
                )
            else:
                tqdm.write(f"Iter <{i}> : cost {cost:.7f} : Δcost N/A")

            previous_cost = cost
        if (i + 1) % save_interval == 0:
            np.save(os.path.join(save_folder, f"W1_iter{i+1}.npy"), W1)
            np.save(os.path.join(save_folder, f"b1_iter{i+1}.npy"), b1)
            np.save(os.path.join(save_folder, f"W2_iter{i+1}.npy"), W2)
            np.save(os.path.join(save_folder, f"b2_iter{i+1}.npy"), b2)
            np.save(os.path.join(save_folder, f"W3_iter{i+1}.npy"), W3)
            np.save(os.path.join(save_folder, f"b3_iter{i+1}.npy"), b3)
            tqdm.write(f"Weights and biases saved at iteration {i+1}.")

        if i % 100 == 0:
            costs.append(cost)

        iter_start_time = time.time()

        iter_start_time = time.time()  # Reset iteration timer

    end_time = time.time()
    training_time = end_time - start_time
    tqdm.write(f"Training completed in {training_time:.2f} seconds.")

    # Save weights and biases after training
    np.save("W1.npy", W1)
    np.save("b1.npy", b1)
    np.save("W2.npy", W2)
    np.save("b2.npy", b2)
    np.save("W3.npy", W3)
    np.save("b3.npy", b3)
    tqdm.write("Weights and biases saved.")

    plt.plot(range(0, epochs, 100), costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Over Iterations')
    plt.savefig(image_path_cost,bbox_inches='tight', dpi=100)
    plt.close()

    return W1, b1, W2, b2, W3, b3, training_time
def gradient_descent(X, y, W1, b1, W2, b2, W3, b3, learning_rate=0.01, epochs=1000):
    save_folder = "checkpoints"
    os.makedirs(save_folder, exist_ok=True)
    previous_cost = None
    image_path_cost = "/root/aiRoot/0-AI/AI/multiClassClassification/png"
    save_interval = 500
    start_time = time.time()
    iter_start_time = time.time()  # To track time per iteration

    costs = []  # To store costs for plotting every 100 iterations

    for i in tqdm(range(epochs), desc="Training Progress", position=0, leave=True):
        z1, a1, z2, a2, a3 = forward_pass(X, W1, b1, W2, b2, W3, b3)
        cost = compute_cost(y, a3)

        # Collect costs every 100 iterations for plotting
        if i % 100 == 0:
            costs.append(cost)

        dW1, db1, dW2, db2, dW3, db3 = compute_gradients(X, y, W1, W2, W3, z1, z2, a1, a2, a3)

        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        W3 -= learning_rate * dW3
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        b3 -= learning_rate * db3

        # Calculate elapsed time and estimated remaining time
        total_elapsed_time = time.time() - start_time
        remaining_iters = epochs - (i + 1)  # Remaining iterations
        avg_time_per_iter = total_elapsed_time / (i + 1)
        estimated_completion_time = remaining_iters * avg_time_per_iter

        # Print status
        if i % 10 == 0:
            if previous_cost is not None:
                delta_cost = previous_cost - cost
                tqdm.write(
                    f"Iter <{i:5d}> : cost {cost:.7f} : Δcost {delta_cost: .7f} | "
                    f"ETC: {format_time(estimated_completion_time):>8} | T: {format_time(total_elapsed_time):>8}"
                )
            else:
                tqdm.write(f"Iter <{i}> : cost {cost:.7f} : Δcost N/A | T: {format_time(total_elapsed_time):>8}")

            previous_cost = cost

        # Save weights and biases every 500 iterations
        if (i + 1) % save_interval == 0:
            np.save(os.path.join(save_folder, f"W1_iter{i+1}.npy"), W1)
            np.save(os.path.join(save_folder, f"b1_iter{i+1}.npy"), b1)
            np.save(os.path.join(save_folder, f"W2_iter{i+1}.npy"), W2)
            np.save(os.path.join(save_folder, f"W2_iter{i+1}.npy"), b2)
            np.save(os.path.join(save_folder, f"W3_iter{i+1}.npy"), W3)
            np.save(os.path.join(save_folder, f"b3_iter{i+1}.npy"), b3)
            tqdm.write(f"Weights and biases saved at iteration {i+1}.")

        iter_start_time = time.time()  # Reset iteration timer

    end_time = time.time()
    training_time = end_time - start_time
    tqdm.write(f"Training completed in {training_time:.2f} seconds.")

    # Save final weights and biases after training
    np.save("models/W1.npy", W1)
    np.save("models/b1.npy", b1)
    np.save("models/W2.npy", W2)
    np.save("models/b2.npy", b2)
    np.save("models/W3.npy", W3)
    np.save("models/b3.npy", b3)
    tqdm.write("Final weights and biases saved.")

    # Plot cost
    plt.plot(range(0, epochs, 100), costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Over Iterations (Every 100 Iterations)')
    plt.savefig(image_path_cost, bbox_inches='tight', dpi=100)
    plt.close()

    return W1, b1, W2, b2, W3, b3, training_time



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


def sample_data_image(rows=10, cols=10):
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


def model_summary(W1, b1, W2, b2, W3, b3, training_time, initial_cost, final_cost, learning_rate, num_epochs, activation_hidden, init_method, accuracy, x_train, y_train):
    # Calculate total parameters
    total_parameters = (W1.size + b1.size) + (W2.size + b2.size) + (W3.size + b3.size)
    cost_reduction = ((initial_cost - final_cost) / initial_cost) * 100  # Percentage reduction

    # Memory usage calculation
    memory_usage_bytes = W1.nbytes + b1.nbytes + W2.nbytes + b2.nbytes + W3.nbytes + b3.nbytes
    activation_memory_bytes = (x_train.nbytes +
                               W1.shape[1] * x_train.shape[0] * 8 +  # a1
                               W2.shape[1] * x_train.shape[0] * 8 +  # a2
                               output_units * x_train.shape[0] * 8)  # a3
    gradient_memory_bytes = memory_usage_bytes  # Roughly equal to weights and biases
    total_memory_usage_bytes = memory_usage_bytes + activation_memory_bytes + gradient_memory_bytes
    total_memory_usage_mb = total_memory_usage_bytes / (1024 ** 2)  # Convert to MB

    # Count images for each label
    unique, counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
    label_counts = dict(zip(unique, counts))

    print("\nModel Summary:")
    print(f"- Total Training Samples: {x_train.shape[0]}")
    for label, count in label_counts.items():
        print(f"- Number of '{label}' Images: {count}")

    print(f"\nTraining Configuration:")
    print(f"- Activation Function (Hidden Layers): {activation_hidden.capitalize()}")
    print(f"- Activation Function (Output Layer): Softmax")
    print(f"- Weight Initialization: {init_method.capitalize()}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Number of Epochs: {num_epochs}")
    print(f"- Initial Cost Value: {initial_cost:.4f}")
    print(f"- Final Cost Value: {final_cost:.4f}")
    print(f"- Cost Reduction: {cost_reduction:.2f}%")
    print(f"- Accuracy: {accuracy:.5f}%")
    print(f"- Training Time: {training_time} \n")

    print("Weight and Bias Parameters:")
    print(f"- W1: {W1.shape}, b1: {b1.shape}")
    print(f"- W2: {W2.shape}, b2: {b2.shape}")
    print(f"- W3: {W3.shape}, b3: {b3.shape}")
    print(f"- Total Parameters: {total_parameters}")
    print(f"- Space Used by Weights and Biases: {memory_usage_bytes / 1024:.2f} KB")
    print(f"- Estimated Total Memory Usage During Training: {total_memory_usage_mb:.2f} MB\n")

    print("Environment Details:")
    print("- Hardware: CPU")  # Update this if GPU is not used
    print("- Python Version:", sys.version)
    print("- NumPy Version:", np.__version__)


def calculate_accuracy(X, y, W1, b1, W2, b2, W3, b3):
    _, _, _, _, a3 = forward_pass(X, W1, b1, W2, b2, W3, b3)
    predictions = np.argmax(a3, axis=1)
    true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == true_labels) * 100
    return accuracy


def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"


def preprocess_custom_image(image_path):
    # Open the image
    img = Image.open(image_path).convert('L')  # Grayscale
    img = ImageOps.invert(img)  # Invert colors if background is white and digit is black
    img = img.resize((28, 28), Image.LANCZOS)  # Resize to 28x28 with smoothing

    # Normalize and flatten
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.reshape(1, -1)  # Flatten for model input
    return img_array
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

weights_and_biases = ["models/W1.npy", "models/b1.npy", "models/W2.npy", "models/b2.npy", "models/W3.npy", "models/b3.npy"]
y_train_one_hot = one_hot_encode(y_train)  # One-hot encode y_train


mode = "t"          # "t" for train, "p" for predict
learning_rate = 0.01
num_epochs = 499


if mode == "t":
    if all(os.path.exists(file) for file in weights_and_biases):
        W1 = np.load("models/W1.npy")
        b1 = np.load("models/b1.npy")
        W2 = np.load("models/W2.npy")
        b2 = np.load("models/b2.npy")
        W3 = np.load("models/W3.npy")
        b3 = np.load("models/b3.npy")
        print("Loaded saved weights and biases ready to continue.")
    else:
        W1 = np.random.randn(input_units, hidden_units_1) * np.sqrt(1 / input_units)
        W2 = np.random.randn(hidden_units_1, hidden_units_2) * np.sqrt(1 / hidden_units_1)
        W3 = np.random.randn(hidden_units_2, output_units) * np.sqrt(1 / hidden_units_2)
        b1 = np.zeros((1, hidden_units_1))
        b2 = np.zeros((1, hidden_units_2))
        b3 = np.zeros((1, output_units))
        print("Initialization complete ready to train.")

    initial_cost = compute_cost(one_hot_encode(y_train), forward_pass(x_train, W1, b1, W2, b2, W3, b3)[-1])
    W1, b1, W2, b2, W3, b3, training_time = gradient_descent(x_train, one_hot_encode(y_train), W1, b1, W2, b2, W3, b3, learning_rate, num_epochs)
    final_cost = compute_cost(one_hot_encode(y_train), forward_pass(x_train, W1, b1, W2, b2, W3, b3)[-1])
    training_time = format_time(training_time)
    np.save("models/W1.npy", W1)
    np.save("models/b1.npy", b1)
    np.save("models/W2.npy", W2)
    np.save("models/b2.npy", b2)
    np.save("models/W3.npy", W3)
    np.save("models/b3.npy", b3)

    accuracy = calculate_accuracy(x_train, one_hot_encode(y_train), W1, b1, W2, b2, W3, b3)
    activation_hidden="ReLu"
    init_method="Xavier"
    model_summary(W1, b1, W2, b2, W3, b3, training_time, initial_cost, final_cost, learning_rate, num_epochs, activation_hidden, init_method, accuracy, x_train, one_hot_encode(y_train))


elif mode == "p":
    W1 = np.load("models/W1.npy")
    b1 = np.load("models/b1.npy")
    W2 = np.load("models/W2.npy")
    b2 = np.load("models/b2.npy")
    W3 = np.load("models/W3.npy")
    b3 = np.load("models/b3.npy")
    print("Loaded saved weights and biases.")

    user_input = input("Enter the range of indices for prediction (e.g., '0-9'): ")
    start, end = map(int, user_input.split('-'))
    sample_images = x_test[start:end + 1]  # Using test data here
    sample_labels = y_test[start:end + 1]  # Using corresponding test labels
    _, _, _, _, softmax_outputs = forward_pass(sample_images.reshape(sample_images.shape[0], -1), W1, b1, W2, b2, W3, b3)
    
    predicted_labels = np.argmax(softmax_outputs, axis=1)

    for i in range(len(predicted_labels)):
        print(f"\nImage Index {start + i}")
        print(f"True Label: {sample_labels[i]} | Predicted Label: {predicted_labels[i]}")
        print("Softmax Probabilities:")
        for digit, prob in enumerate(softmax_outputs[i]):
            prob_percentage = prob * 100
            print(f"  {digit}: {prob_percentage:.9f}% {'<-- Predicted' if digit == predicted_labels[i] else ''}")

    num_samples = len(sample_images)
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    for i in range(num_samples):
        axes[i].imshow(sample_images[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Pred: {predicted_labels[i]}\nTrue: {sample_labels[i]}")

    plt.savefig(image_path,bbox_inches='tight', dpi=100)
    plt.close()

else:
    # sample_data_image(10,10)
    # print("Invalid mode selected.")
    image_path = '/root/aiRoot/0-AI/AI/MultiClassClassification/png/output.png'
    W1 = np.load("models/W1.npy")
    b1 = np.load("models/b1.npy")
    W2 = np.load("models/W2.npy")
    b2 = np.load("models/b2.npy")
    W3 = np.load("models/W3.npy")
    b3 = np.load("models/b3.npy")

    # Predict using the model
    processed_img = preprocess_custom_image(image_path)
    _, _, _, _, predictions = forward_pass(processed_img, W1, b1, W2, b2, W3, b3)

    # Display results
    predicted_digit = np.argmax(predictions)
    predictions = predictions.tolist()
    print(f"Predicted Digit: {predicted_digit}")
    for i in range(len(predictions)):
        print(f"Prediction softmax {i} --> {predictions[i]*100}")
