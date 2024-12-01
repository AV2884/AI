import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z))  
    return exp_z / np.sum(exp_z, axis=0)


def initialize_parameters(input_size, hidden_size1, hidden_size2, output_size):
    params = {
        "Wx1": np.random.randn(hidden_size1, input_size) * np.sqrt(1 / input_size),
        "Wh1": np.random.randn(hidden_size1, hidden_size1) * np.sqrt(1 / hidden_size1),
        "bh1": np.zeros((hidden_size1, 1)),
        "Wx2": np.random.randn(hidden_size2, hidden_size1) * np.sqrt(1 / hidden_size1),
        "Wh2": np.random.randn(hidden_size2, hidden_size2) * np.sqrt(1 / hidden_size2),
        "bh2": np.zeros((hidden_size2, 1)),
        "Wy": np.random.randn(output_size, hidden_size2) * np.sqrt(1 / hidden_size2),
        "by": np.zeros((output_size, 1)),
    }
    return params



def forward_pass_stacked(X, params):
    Wx1, Wh1, bh1 = params["Wx1"], params["Wh1"], params["bh1"]
    Wx2, Wh2, bh2 = params["Wx2"], params["Wh2"], params["bh2"]
    Wy, by = params["Wy"], params["by"]

    sequence_length = X.shape[0]
    hidden_size1 = Wh1.shape[0]
    hidden_size2 = Wh2.shape[0]

    h_t1 = np.zeros((hidden_size1, 1))
    h_t2 = np.zeros((hidden_size2, 1))

    hidden_states_layer1 = []
    hidden_states_layer2 = []

    for t in range(sequence_length):
        x_t = X[t].reshape(-1, 1)
        h_t1 = np.tanh(np.dot(Wx1, x_t) + np.dot(Wh1, h_t1) + bh1)
        hidden_states_layer1.append(h_t1)
        
        h_t2 = np.tanh(np.dot(Wx2, h_t1) + np.dot(Wh2, h_t2) + bh2)
        hidden_states_layer2.append(h_t2)

    y_t = softmax(np.dot(Wy, h_t2) + by)

    return y_t, (hidden_states_layer1, hidden_states_layer2)



def compute_loss(y_pred, y_true):
    """
    Compute cross-entropy loss.
    :param y_pred: Predicted probabilities, shape (output_size, 1).
    :param y_true: True label (integer class).
    :return: Loss value.
    """
    loss = -np.log(y_pred[y_true, 0])
    return loss


def compute_loss_batch(y_pred, y_true):
    m = y_true.shape[0]  # Batch size
    log_probs = -np.log(y_pred[range(m), y_true])  # True class probabilities
    loss = np.sum(log_probs) / m
    return loss


def compute_gradients_stacked(y_pred, y_true, hidden_states, X, params):
    Wx1, Wh1, bh1 = params["Wx1"], params["Wh1"], params["bh1"]
    Wx2, Wh2, bh2 = params["Wx2"], params["Wh2"], params["bh2"]
    Wy, by = params["Wy"], params["by"]

    hidden_states_layer1, hidden_states_layer2 = hidden_states
    sequence_length = X.shape[0]

    dWx1, dWh1, dbh1 = np.zeros_like(Wx1), np.zeros_like(Wh1), np.zeros_like(bh1)
    dWx2, dWh2, dbh2 = np.zeros_like(Wx2), np.zeros_like(Wh2), np.zeros_like(bh2)
    dWy, dby = np.zeros_like(Wy), np.zeros_like(by)

    delta_y = y_pred
    delta_y[y_true] -= 1
    dWy += np.dot(delta_y, hidden_states_layer2[-1].T)
    dby += delta_y

    delta_h2 = np.dot(Wy.T, delta_y)
    for t in reversed(range(sequence_length)):
        h_t2 = hidden_states_layer2[t]
        h_t1 = hidden_states_layer1[t]
        x_t = X[t].reshape(-1, 1)

        # Backprop through layer 2
        delta_h2 = delta_h2 * (1 - h_t2**2)
        dWx2 += np.dot(delta_h2, h_t1.T)
        dWh2 += np.dot(delta_h2, hidden_states_layer2[t - 1].T if t > 0 else np.zeros_like(h_t2).T)
        dbh2 += delta_h2

        delta_h1 = np.dot(Wx2.T, delta_h2)  # Propagate only through Wx2

        # Backprop through layer 1
        delta_h1 = delta_h1 * (1 - h_t1**2)
        dWx1 += np.dot(delta_h1, x_t.T)
        dWh1 += np.dot(delta_h1, hidden_states_layer1[t - 1].T if t > 0 else np.zeros_like(h_t1).T)
        dbh1 += delta_h1

    gradients = {
        "Wx1": dWx1, "Wh1": dWh1, "bh1": dbh1,
        "Wx2": dWx2, "Wh2": dWh2, "bh2": dbh2,
        "Wy": dWy, "by": dby,
    }
    for key, grad in gradients.items():
        print(f"{key} gradient norm: {np.linalg.norm(grad)}")
    return gradients



def back_propogation_stacked(y_pred, y_true, hidden_states, X, params, learning_rate):
    gradients = compute_gradients_stacked(y_pred, y_true, hidden_states, X, params)

    for key in params:
        params[key] -= learning_rate * gradients[key]

    return params



def train_rnn_stacked(X_train, y_train, params, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for X, y in zip(X_train, y_train):
            y_pred, hidden_states = forward_pass_stacked(X, params)
            loss = compute_loss(y_pred, int(y))
            total_loss += loss
            params = back_propogation_stacked(y_pred, int(y), hidden_states, X, params, learning_rate)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train):.7f}")
    return params


# Load the preprocessed data
X_train = np.load("data/final_data/X_train.npy")
y_train = np.load("data/final_data/y_train.npy")

# Hyperparameters
input_size = 7         # Number of features in each time step
hidden_size1 = 32      # Hidden units in the first hidden layer
hidden_size2 = 16      # Hidden units in the second hidden layer
output_size = 3        # Number of output classes (Up, Down, Flat)
learning_rate = 0.01   # Learning rate for gradient descent
epochs = 10            # Number of training epochs

# Initialize parameters
params = initialize_parameters(input_size, hidden_size1, hidden_size2, output_size)

# Train the RNN
params = train_rnn_stacked(X_train, y_train, params, learning_rate, epochs)
