import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z))  
    return exp_z / np.sum(exp_z, axis=0)


def initialize_parameters(input_size, hidden_size, output_size):
    params = {
        "Wx": np.random.randn(hidden_size, input_size) * np.sqrt(1 / input_size),
        "Wh": np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / hidden_size),
        "Wy": np.random.randn(output_size, hidden_size) * np.sqrt(1 / hidden_size),
        "bh": np.zeros((hidden_size, 1)),
        "by": np.zeros((output_size, 1)),
    }
    return params


def forward_pass(X, params):
    """
    :param X: Input sequence of shape (sequence_length, input_size).
    :param params: Dictionary containing RNN parameters.
    :return: Final output (y_t), all hidden states, and cache for backpropagation.
    """
    Wx, Wh, Wy, bh, by = params["Wx"], params["Wh"], params["Wy"], params["bh"], params["by"]
    sequence_length, input_size = X.shape
    hidden_size = Wh.shape[0]

    h_t = np.zeros((hidden_size,1))
    hidden_states = []
    
    for t in range(sequence_length):
        x_t = X[t].reshape(-1, 1)  # Reshape to (input_size, 1)
        h_t = np.tanh(np.dot(Wx, x_t) + np.dot(Wh, h_t) + bh)
        hidden_states.append(h_t)
    
    y_t = softmax(np.dot(Wy, h_t) + by)

    return y_t, hidden_states


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


def compute_gradients(y_pred, y_true, hidden_states, X, params):
    """
    Compute gradients for RNN parameters using BPTT.
    :param y_pred: Predicted probabilities, shape (output_size, 1).
    :param y_true: True label (integer class).
    :param hidden_states: List of hidden states for all time steps.
    :param X: Input sequence, shape (sequence_length, input_size).
    :param params: Dictionary of RNN parameters.
    :return: Gradients for all parameters.
    """
    Wx, Wh, Wy, bh, by = params["Wx"], params["Wh"], params["Wy"], params["bh"], params["by"]
    sequence_lenght = X.shape[0]
    hidden_size = Wh.shape[0]

    # Initialize gradients
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    # Compute output error (delta_y)
    delta_y = y_pred
    delta_y[y_true] -= 1
    dWy += np.dot(delta_y, hidden_states[-1].T)  # Gradient of Wy
    dby += delta_y                               # Gradient of by

    delta_h = np.dot(Wy.T, delta_y)              # Initialize delta_h for the last time step

    # Backpropagate through time
    for t in reversed(range(sequence_lenght)):
        h_t = hidden_states[t]
        x_t = X[t].reshape(-1,1)

        delta_h = delta_h * (1 - h_t**2)         # Backpropagate through tanh 

        # Gradients for Wx, Wh, bh
        dWx += np.dot(delta_h, x_t.T)
        dWh += np.dot(delta_h, hidden_states[t - 1].T if t > 0 else np.zeros_like(h_t).T)
        dbh += delta_h

        delta_h = np.dot(Wh.T, delta_h)           # Propogate delta_h backwards

        gradients = {"Wx": dWx, "Wh": dWh, "Wy": dWy, "bh": dbh, "by": dby}

    return gradients



def back_propogation(y_pred, y_true, hidden_states, X, params, learning_rate):

    #Compute gradients
    gradients = compute_gradients(y_pred, y_true, hidden_states, X, params)

    #Update parameters
    params["Wx"] -= learning_rate * gradients["Wx"]
    params["Wh"] -= learning_rate * gradients["Wh"]
    params["Wy"] -= learning_rate * gradients["Wy"]
    params["bh"] -= learning_rate * gradients["bh"]
    params["by"] -= learning_rate * gradients["by"]

    return params



def train_rnn(X_train, y_train, params, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for X, y in zip(X_train, y_train):  # Ensure zip matches shapes
            y_pred, hidden_states = forward_pass(X, params)

            # Convert label to integer if necessary
            loss = compute_loss(y_pred, int(y))  # Ensure y is an integer
            total_loss += loss

            # Backpropagation
            params = back_propogation(y_pred, int(y), hidden_states, X, params, learning_rate)

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train):.7f}")
    return params


X_train = np.load("data/final_data/X_train.npy")
y_train = np.load("data/final_data/y_train.npy")

input_size = 7
hidden_size = 16
output_size = 3
learning_rate = 0.01
epochs = 100

params = initialize_parameters(input_size, hidden_size, output_size)
params = train_rnn(X_train, y_train, params, learning_rate, epochs)
