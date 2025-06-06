import numpy as np

# ====================== CONFIG ======================
d_model = 2  # dimensionality of embeddings and hidden vectors
sequence = [1.0, 2.0, 3.0]
positions = [1, 2, 3]

# ====================== STEP 1: EMBEDDING ======================
print("\n🔹 STEP 1: Embedding Input Numbers into Vectors")

W = np.array([[0.1], [0.3]])       # shape: (2, 1)
b = np.array([0.01, 0.02])         # shape: (2,)

embedded_sequence = []
for x in sequence:
    x_vector = np.array([[x]])                     # shape: (1,1)
    embed = W @ x_vector + b.reshape(-1, 1)        # shape: (2,1)
    embedded_sequence.append(embed.flatten())      # flatten to shape: (2,)

embedded = np.array(embedded_sequence)

print("\n🔸 Scalar Inputs → Embedded Vectors (Wx + b):")
for i, vec in enumerate(embedded):
    print(f"Token {sequence[i]:.1f} → {vec}")

# ====================== STEP 2: POSITIONAL ENCODING ======================
print("\n🔹 STEP 2: Add Positional Encoding (sin/cos)")

def simple_positional_encoding(positions):
    return np.array([[np.sin(pos), np.cos(pos)] for pos in positions])

position_encoding = simple_positional_encoding(positions)
final_input = embedded + position_encoding

print("\n🔸 Positional Encodings:")
for i, vec in enumerate(position_encoding):
    print(f"Position {positions[i]} → {vec}")

print("\n🔸 Final Input to Transformer (Embedding + Position):")
for i, vec in enumerate(final_input):
    print(f"Token {i} → {vec}")

# ====================== STEP 3: SELF-ATTENTION ======================
print("\n🔹 STEP 3: Self-Attention Mechanism")

Wq = np.array([[0.1, 0.2], [0.3, 0.4]])
Wk = np.array([[0.2, 0.1], [0.4, 0.3]])
Wv = np.array([[1.0, 0.0], [0.0, 1.0]])  # identity matrix

# Q, K, V projections
Q = final_input @ Wq.T
K = final_input @ Wk.T
V = final_input @ Wv.T

print("\n🔸 Query Vectors (Q):\n", Q)
print("🔸 Key Vectors (K):\n", K)
print("🔸 Value Vectors (V):\n", V)

# Compute attention scores (Q · Kᵗ)
scores = Q @ K.T
scaled_scores = scores / np.sqrt(d_model)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)

print("\n🔸 Scaled Attention Scores:\n", scaled_scores)
print("\n🔸 Attention Weights (Softmax):\n", attention_weights)

# Weighted sum of values
attention_output = attention_weights @ V

print("\n🔸 Self-Attention Output (Weighted Sum of V):")
for i, vec in enumerate(attention_output):
    print(f"Token {i} → {vec}")

# ====================== STEP 4: FEEDFORWARD NETWORK ======================
print("\n🔹 STEP 4: Feedforward Layer (2 → 4 → ReLU → 2)")

W1 = np.array([[0.5, -0.2],
               [-0.3, 0.8],
               [0.1, 0.6],
               [0.7, -0.5]])    # shape: (4, 2)

b1 = np.array([0.1, 0.2, 0.3, 0.4])  # shape: (4,)

W2 = np.array([[0.2, -0.4, 0.1, 0.3],
               [0.5, 0.1, -0.2, 0.6]])  # shape: (2, 4)

b2 = np.array([0.05, -0.05])  # shape: (2,)

def relu(x):
    return np.maximum(0, x)

feedforward_output = []
for vec in attention_output:
    h = W1 @ vec + b1
    h_relu = relu(h)
    out = W2 @ h_relu + b2
    feedforward_output.append(out)

feedforward_output = np.array(feedforward_output)

print("\n🔸 Output After Feedforward (per token):")
for i, vec in enumerate(feedforward_output):
    print(f"Token {i} → {vec}")

# ====================== STEP 5: FINAL PREDICTION ======================
print("\n🔹 STEP 5: Final Prediction (Reduce to Scalar)")

# Combine all tokens into a single vector (mean)
final_representation = feedforward_output.mean(axis=0)
print(f"\n🔸 Combined Vector (Mean of Tokens): {final_representation}")

# Final linear layer to predict next number
W_out = np.array([[1.0, 0.5]])   # shape: (1, 2)
b_out = np.array([0.1])          # shape: (1,)

pred = W_out @ final_representation + b_out
pred_scalar = pred.item()

print(f"\n🎯 Final Prediction for next number: {pred_scalar:.4f}")
