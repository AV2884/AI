import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CustomRNNCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.hidden_size = hidden_size

    @property
    def state_size(self):
        return self.hidden_size

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.Wx = self.add_weight(shape=(input_dim, self.hidden_size), initializer="random_normal", trainable=True)
        self.Wh = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True)

    def call(self, inputs, states):
        prev_hidden = states[0]
        current_hidden = tf.nn.tanh(tf.matmul(inputs, self.Wx) + tf.matmul(prev_hidden, self.Wh) + self.b)
        return current_hidden, [current_hidden]


class StackedRNNModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(StackedRNNModel, self).__init__()
        self.rnn1 = tf.keras.layers.RNN(CustomRNNCell(hidden_size1), return_sequences=True, return_state=True)
        self.rnn2 = tf.keras.layers.RNN(CustomRNNCell(hidden_size2), return_state=True)
        self.dense = tf.keras.layers.Dense(output_size, activation="softmax")

    def call(self, inputs):
        rnn1_output, *rnn1_state = self.rnn1(inputs)
        rnn2_output, *rnn2_state = self.rnn2(rnn1_output)
        output = self.dense(rnn2_output)
        return output


def train_model(X_train, y_train, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs):
    model = StackedRNNModel(input_size, hidden_size1, hidden_size2, output_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = []
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in zip(X_train, y_train):
            with tf.GradientTape() as tape:
                X_batch = tf.expand_dims(X_batch, axis=0)  # Add batch dimension
                y_pred = model(X_batch)
                loss = loss_fn([y_batch], y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / len(X_train)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, train_loss


def predict(model, X_test):
    predictions = []
    for X_batch in X_test:
        X_batch = tf.expand_dims(X_batch, axis=0)  # Add batch dimension
        y_pred = model(X_batch)
        predictions.append(tf.argmax(y_pred, axis=-1).numpy()[0])
    return predictions


if __name__ == "__main__":
    # Load your CSV data
    df = pd.read_csv("your_csv_file.csv")
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values  # Labels

    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Split into train and test
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Model parameters
    input_size = X_train.shape[1]
    hidden_size1 = 32
    hidden_size2 = 16
    output_size = len(np.unique(y_train))  # Number of unique labels
    learning_rate = 0.001
    epochs = 10

    # Train model
    model, train_loss = train_model(X_train, y_train, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs)

    # Plot training loss
    plt.plot(range(1, epochs + 1), train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    # Predictions
    predictions = predict(model, X_test)
    print(f"Predictions: {predictions}")
