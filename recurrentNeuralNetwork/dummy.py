#Param init:
input_size = 7
hidden_size = 16
output_size = 3
params = initialize_parameters(input_size, hidden_size, output_size)
for key, value in params.items():
    print(f"{key}: {value.shape}")

#forward_pass:
X = np.random.randn(5, 7)
y_t, hidden_states = forward(X, params)
print(f"Output (y_t): {y_t.shape}")
print(f"Hidden state 1 (h_t): {hidden_states[0].shape}")
print(f"Hidden state count: {len(hidden_states)}")

#loss:
y_pred = np.array([[1], [1], [1]])  # Predicted probabilities
y_true = 1  # True label (class index)
loss = compute_loss(y_pred, y_true)
print(f"Loss: {loss}")