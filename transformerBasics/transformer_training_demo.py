import numpy as np

#==================== CONFIG ====================
d_model = 2
sequence = [4.0, 3.0, 2.0]
positions = [1, 2, 3]
target = 1.0
lr = 0.01
epochs = 1000

#==================== HELPER FUNCTIONS ====================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def simple_positional_encoding(positions):
    return np.array([[np.sin(pos), np.cos(pos)] for pos in positions])

#==================== FIXED TRANSFORMER BLOCK ====================
# (non-trainable for now — frozen backbone)

# Embedding weights
W_embed = np.array([[0.1], [0.3]])
b_embed = np.array([0.01, 0.02])

# Positional encoding
pos_encoding = simple_positional_encoding(positions)

# Q, K, V weights
Wq = np.array([[0.1, 0.2], [0.3, 0.4]])
Wk = np.array([[0.2, 0.1], [0.4, 0.3]])
Wv = np.array([[1.0, 0.0], [0.0, 1.0]])

# Feedforward weights
W1 = np.array([[0.5, -0.2], [-0.3, 0.8], [0.1, 0.6], [0.7, -0.5]])
b1 = np.array([0.1, 0.2, 0.3, 0.4])
W2 = np.array([[0.2, -0.4, 0.1, 0.3], [0.5, 0.1, -0.2, 0.6]])
b2 = np.array([0.05, -0.05])

#==================== TRAINABLE OUTPUT LAYER ====================
np.random.seed(42)
W_out = np.random.randn(1, d_model)
b_out = np.zeros((1,))

#==================== TRAINING LOOP ====================
for epoch in range(epochs):
    # ----- Step 1: Embedding -----
    embedded = []
    for x in sequence:
        x_vec = np.array([[x]])
        emb = W_embed @ x_vec + b_embed.reshape(-1, 1)
        embedded.append(emb.flatten())
    embedded = np.array(embedded)

    # ----- Step 2: Add Positional Encoding -----
    final_input = embedded + pos_encoding

    # ----- Step 3: Self-Attention -----
    Q = final_input @ Wq.T
    K = final_input @ Wk.T
    V = final_input @ Wv.T
    scores = Q @ K.T
    scaled_scores = scores / np.sqrt(d_model)
    attention_weights = softmax(scaled_scores)
    attention_output = attention_weights @ V

    # ----- Step 4: Feedforward -----
    ff_output = []
    for vec in attention_output:
        h = W1 @ vec + b1
        h_relu = relu(h)
        out = W2 @ h_relu + b2
        ff_output.append(out)
    ff_output = np.array(ff_output)  # shape (3,2)

    # ----- Step 5: Final Prediction Head -----
    x_avg = ff_output.mean(axis=0)           # shape (2,)
    y_pred = W_out @ x_avg + b_out           # shape (1,)
    pred_scalar = y_pred.item()
    loss = (pred_scalar - target) ** 2

    # ----- Step 6: Backprop (for final layer only) -----
    dloss = 2 * (pred_scalar - target)
    dW_out = dloss * x_avg.reshape(1, -1)
    db_out = dloss

    # ----- Step 7: Update -----
    W_out -= lr * dW_out
    b_out -= lr * db_out

    # ----- Logging -----
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:>4} | Prediction: {pred_scalar:.4f} | Loss: {loss:.6f}")

