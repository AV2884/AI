import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from kiteconnect import KiteConnect

# --- KiteConnect Setup ---
API_KEY = "y1bems9r9oo388js"
API_SECRET = "1qd5jtc2xbwsx2dn2c7q3hulsvp74hxp"
ACCESS_TOKEN = "Oj58x9ZPQADrD60yrZTrsigckZIYL6FT"
NFT = 256265  # Nifty 50
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# --- Data Fetching ---
def fetch_historical_data(instrument_token, from_date, to_date, interval):
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
    )
    return pd.DataFrame(data)

# --- Save Data ---
def save_to_csv(df, instrument_token):
    file_path = "data/NFT_raw_one_day.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Data fetching complete. Saved to {file_path}")

# --- Preprocessing ---
def normalize_data(df, columns):
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def generate_labels(df, threshold=0.5):
    pct_change = df["close"].pct_change() * 100
    labels = []
    for change in pct_change:
        if change > threshold:
            labels.append(2)  # Up
        elif change < -threshold:
            labels.append(0)  # Down
        else:
            labels.append(1)  # Flat
    df["label"] = labels
    return df.iloc[1:]

def create_sequences(df, sequence_length, target_column):
    inputs, outputs = [], []
    for i in range(len(df) - sequence_length):
        inputs.append(
            df.iloc[i:i + sequence_length][["open", "high", "low", "close", "hour", "day_of_week"]].values
        )
        outputs.append(df.iloc[i + sequence_length][target_column])
    return np.array(inputs), np.array(outputs)

def preprocess_latest_data():
    raw_data = pd.read_csv("data/NFT_raw_one_day.csv")
    raw_data["date"] = pd.to_datetime(raw_data["date"])
    raw_data["hour"] = raw_data["date"].dt.hour
    raw_data["day_of_week"] = raw_data["date"].dt.dayofweek
    raw_data = raw_data.drop(columns=["date", "volume"])
    normalized_data = normalize_data(raw_data, ["open", "high", "low", "close"])
    labeled_data = generate_labels(normalized_data, threshold=0.5)
    X_test, y_test = create_sequences(labeled_data, sequence_length=5, target_column="label")
    return X_test, y_test, labeled_data["close"].values

# --- Model Functions ---
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0)

def forward_pass_stacked(X, params):
    Wx1, Wh1, bh1 = params["Wx1"], params["Wh1"], params["bh1"]
    Wx2, Wh2, bh2 = params["Wx2"], params["Wh2"], params["bh2"]
    Wy, by = params["Wy"], params["by"]

    sequence_length = X.shape[0]
    h_t1 = np.zeros((Wx1.shape[0], 1))
    h_t2 = np.zeros((Wx2.shape[0], 1))

    for t in range(sequence_length):
        x_t = X[t].reshape(-1, 1)
        h_t1 = np.tanh(np.dot(Wx1, x_t) + np.dot(Wh1, h_t1) + bh1)
        h_t2 = np.tanh(np.dot(Wx2, h_t1) + np.dot(Wh2, h_t2) + bh2)

    y_t = softmax(np.dot(Wy, h_t2) + by)
    return y_t

def load_model(file_path="model.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            params = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return params
    else:
        print("No saved model found.")
        return None

def predict(X, params):
    y_pred = forward_pass_stacked(X, params)
    return np.argmax(y_pred, axis=0).item()

# --- Simulation ---
def plot_predictions(prices, predictions, actual_labels):
    plt.figure(figsize=(10, 6))
    plt.plot(prices, label="Actual Prices", color="blue", marker="o")
    for i in range(len(predictions)):
        color = "green" if predictions[i] == actual_labels[i] else "red"
        plt.scatter(i + 1, prices[i + 1], color=color)
    plt.title("Price Predictions vs Actual Movements")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend(["Prices", "Correct Prediction", "Wrong Prediction"], loc="upper left")
    plt.grid()
    plt.show()

def simulate_latest_day(X_test, y_test, params, close_prices):
    correct = 0
    predictions = []

    print("Simulated Predictions for the Latest Day:")
    print("-" * 50)

    for i, (X, y) in enumerate(zip(X_test, y_test)):
        y_pred = predict(X, params)
        predictions.append(y_pred)

        actual_price = close_prices[i]
        prediction_label = ["Down", "Flat", "Up"][y_pred]
        actual_label = ["Down", "Flat", "Up"][int(y)]

        is_correct = "Correct!" if y_pred == int(y) else "Wrong!"
        print(f"Step {i + 1}: Price {actual_price:.2f} | Pred: {prediction_label} | Act: {actual_label} | {is_correct}")

        if y_pred == int(y):
            correct += 1

    accuracy = (correct / len(y_test)) * 100
    print("-" * 50)
    print(f"Simulated Accuracy: {accuracy:.2f}%")

    plot_predictions(close_prices[:len(predictions) + 1], predictions, y_test)

# --- Main ---
if __name__ == "__main__":
    DAYS = 1
    INTERVAL = "minute"
    instrument_to_fetch = NFT

    to_date = datetime.now()
    from_date = to_date - timedelta(days=DAYS)
    print(f"Fetching data from {from_date} to {to_date}...")

    data = fetch_historical_data(
        instrument_token=instrument_to_fetch,
        from_date=from_date,
        to_date=to_date,
        interval=INTERVAL,
    )
    save_to_csv(data, instrument_to_fetch)

    X_test, y_test, close_prices = preprocess_latest_data()
    params = load_model()

    if params:
        simulate_latest_day(X_test, y_test, params, close_prices)
