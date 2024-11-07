import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Define softmax and relu functions
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # Stability improvement
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

# Forward pass function
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, a3

# Load weights and biases
W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")
W3 = np.load("W3.npy")
b3 = np.load("b3.npy")

# Streamlit app layout
st.title("Draw a Digit (28x28) and Predict!")

# Drawing canvas for user input
canvas = st_canvas(
    fill_color="#000000",  # Background color
    stroke_width=10,       # Stroke width
    height=280,            # Canvas height
    width=280,             # Canvas width
    drawing_mode="freedraw",  # Free drawing mode
    key="canvas",
)
if canvas.image_data is not None:
    # Get the alpha channel (transparency) to detect drawn areas
    img_alpha = canvas.image_data[:, :, 3]  # Alpha channel: 0 (transparent) to 1 (opaque)

    # Scale alpha values to 255, invert them (for black on white)
    img_alpha_inverted = (1 - img_alpha) * 255

    # Convert to grayscale
    img = Image.fromarray(img_alpha_inverted.astype("uint8"))

    # Resize to 28x28 and normalize
    img = img.resize((28, 28)).convert("L")
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_flat = img_array.flatten().reshape(1, -1)

    # Display the processed image for verification
    st.image(img, caption="Processed Input (28x28)", use_column_width=False)

    # Add a "Predict" button
    if st.button("Predict"):
        _, _, _, _, softmax_probs = forward_pass(img_flat, W1, b1, W2, b2, W3, b3)
        prediction = np.argmax(softmax_probs)

        st.write(f"Prediction: {prediction}")
        st.bar_chart(softmax_probs[0])
