import tensorflow as tf
import numpy as np

# Load the trained model
def load_trained_model(model_path="rnn_model_tf.h5"):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

# Make predictions
def make_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)  # Get the class with the highest probability
    return predicted_classes

# Evaluate Predictions
def evaluate_predictions(predicted_classes, y_test):
    accuracy = np.mean(predicted_classes == y_test) * 100
    print(f"Prediction Accuracy: {accuracy:.2f}%")
    return accuracy

# Visualization
def visualize_predictions(predicted_classes, y_test):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Labels", color="blue", marker="o")
    plt.plot(predicted_classes, label="Predicted Labels", color="green", linestyle="dashed", marker="x")
    plt.xlabel("Sample Index")
    plt.ylabel("Class")
    plt.title("Actual vs Predicted Labels")
    plt.legend()
    plt.grid()
    plt.savefig("predictions_comparison.png")
    plt.show()
    print("Prediction comparison graph saved as 'predictions_comparison.png'.")

if __name__ == "__main__":
    # Load Data
    X_test = np.load("/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/processed/X_test.npy")
    y_test = np.load("/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/processed/y_test.npy")
    # Load Model
    model = load_trained_model("rnn_model_tf.h5")

    # Make Predictions
    predicted_classes = make_predictions(model, X_test, y_test)
    print(f"Predicted Classes: {predicted_classes[:10]}")  # Print first 10 predictions
    print(f"Actual Classes: {y_test[:10]}")  # Print first 10 actual labels

    # Evaluate Predictions
    accuracy = evaluate_predictions(predicted_classes, y_test)

    # Visualize Predictions
    visualize_predictions(predicted_classes, y_test)
