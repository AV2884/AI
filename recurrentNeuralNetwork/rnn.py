import numpy as np

def initialize_parameters(input_size, hidden_size, output_size):
    params = {
        "Wx": np.random.randn(hidden_size, input_size) * 0.01,  
        "Wh": np.random.randn(hidden_size, hidden_size) * 0.01, 
        "Wy": np.random.randn(output_size, hidden_size) * 0.01, 
        "bh": np.zeros((hidden_size, 1)),  
        "by": np.zeros((output_size, 1)),  # Output bia
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
    
    y_t = np.dot(Wy , h_t) + by

    return y_t, hidden_states


def compute_loss(y_pred, y_true):
    """
    Compute cross-entropy loss.
    :param y_pred: Predicted probabilities, shape (output_size, 1).
    :param y_true: True label (integer class).
    :return: Loss value.
    """
    m = y_pred.shape[1]  
    log_probs = -np.log(y_pred[y_true, 0])  #This is a vector of predicted probabilities for all classes (e.g., Down, Flat, Up).
    loss = np.sum(log_probs) / m
    return loss


def compute_gradient(y_pred, y_true, hidden_states, X, params):
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


#Dummy code:--------------------------------------------------------
input_size = 7
hidden_size = 16
output_size = 3
params = initialize_parameters(input_size, hidden_size, output_size)
for key, value in params.items():
    print(f"{key}: {value.shape}")

X = np.random.randn(5, 7)
y_t, hidden_states = forward_pass(X, params)
print(f"Output (y_t): {y_t.shape}")
print(f"Hidden state 1 (h_t): {hidden_states[0].shape}")
print(f"Hidden state count: {len(hidden_states)}")

y_pred = np.array([[1], [1], [1]])  # Predicted probabilities
y_true = 1  # True label (class index)
loss = compute_loss(y_pred, y_true)
print(f"Loss: {loss}")
