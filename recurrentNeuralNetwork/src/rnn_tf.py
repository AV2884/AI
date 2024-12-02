import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import os

# Load Data
def load_data():
    
    X_train = np.load("/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/processed/X_train.npy")
    y_train = np.load("/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/processed/y_train.npy")
    X_test = np.load("/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/processed/X_test.npy")
    y_test = np.load("/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/processed/y_test.npy")
    return X_train, y_train, X_test, y_test

# Define the TensorFlow RNN Model
def create_tf_rnn(input_size, hidden_size1, hidden_size2, output_size):
    model = Sequential([
        LSTM(hidden_size1, return_sequences=True, input_shape=(None, input_size)),
        LSTM(hidden_size2),
        Dense(output_size, activation='softmax')
    ])
    return model

# Train the TensorFlow RNN Model
def train_tf_rnn(model, X_train, y_train, learning_rate, epochs, batch_size):
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Plot training loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.plot(history.history['accuracy'], label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Training Metrics')
    plt.grid()
    plt.savefig("training_metrics.png")

    return model

# Evaluate the Model
def evaluate_tf_rnn(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy

# Make Predictions
def predict_tf_rnn(model, X_test):
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

# Main function
if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    print("Data Loaded:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Model Configuration
    input_size = X_train.shape[2]  # Number of features
    hidden_size1 = 32
    hidden_size2 = 16
    output_size = 3  # Classes: Down, Flat, Up
    learning_rate = 0.001
    epochs = 100
    batch_size = 32

    # Create, Train, and Evaluate the Model
    model = create_tf_rnn(input_size, hidden_size1, hidden_size2, output_size)
    model.summary()
    model = train_tf_rnn(model, X_train, y_train, learning_rate, epochs, batch_size)
    evaluate_tf_rnn(model, X_test, y_test)

    # Save Model
    model.save("rnn_model_tf.h5")
    print("Model saved to 'rnn_model_tf.h5'.")

    # Predict
    predictions = predict_tf_rnn(model, X_test)
    print(f"Predictions: {predictions[:10]}")  # Show first 10 predictions
