import pandas as pd
from tqdm import tqdm
import sys
import numpy as np


def validation_save(X, y, filename_prefix):
    flattened_X = X.reshape(X.shape[0], -1)

    df = pd.DataFrame(flattened_X)
    df['label'] = y  # Add labels as the last column

    output_file = f"data/{filename_prefix}_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {filename_prefix} data to {output_file}")


def load_csv(instrument_to_proccess):
    base_path = "/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/"
    file_path = base_path + f"{instrument_to_proccess}_raw.csv"
    try:
        data = pd.read_csv(file_path)
        print("Loaded data")
        return data
    except:
        print("Could not load data")
        sys.exit(0)


def normalize_data(df, columns):
    print("Normalizing data:")
    for col in columns:
        print(f"     Normalizing -> {col}")
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    print("Normalization complete:")
    print(df.head(10))
    return df


def generate_labels(df, threshold=0.5):
    # print(f"Generating labels with threshold {threshold}")

    # pct_change = df['close'].pct_change() * 100
    # labels = []

    # for change in pct_change:
    #     if change > threshold:
    #         labels.append(2)
    #     elif change < -threshold:
    #         labels.append(0)
    #     else:
    #         labels.append(1)
    # df['label'] = labels
    # return df.iloc[1:]
    print(f"Generating labels with threshold {threshold}")

    pct_change = df['close'].pct_change() * 100
    labels = []

    # Use tqdm to add a progress bar
    for change in tqdm(pct_change, desc="Generating labels"):
        if change > threshold:
            labels.append(2)
        elif change < -threshold:
            labels.append(0)
        else:
            labels.append(1)
    print("Updating DF....")
    df['label'] = labels
    return df.iloc[1:]


def create_sequences(df, sequence_length, target_column):
    inputs = []
    outputs = []
    for i in range(len(df) - sequence_length):
        # Include 'hour', 'day_of_week', and 'minute_of_day' in input features
        inputs.append(
            df.iloc[i:i + sequence_length][
                ['open', 'high', 'low', 'close', 'hour', 'day_of_week', 'minute_of_day']
            ].values
        )
        outputs.append(df.iloc[i + sequence_length][target_column])
    return np.array(inputs), np.array(outputs)


if __name__ == "__main__":
    # ------------------------------------------------
    instrument_to_proccess = "NFT"
    threshold = 0.5
    sequence_length = 5  # Number of time steps in each input sequence
    # ------------------------------------------------
    data = load_csv(instrument_to_proccess)
    print(data.head())
    print("- - - Ready to process - - -")

    # Step 1: Parse datetime and calculate time-based features
    data['date'] = pd.to_datetime(data['date'])
    data['hour'] = data['date'].dt.hour
    data['minute'] = data['date'].dt.minute
    # Calculate "Minute of Day" (minutes since 9:15 AM)
    data['minute_of_day'] = (data['hour'] - 9) * 60 + data['minute'] - 15
    data['minute_of_day'] = data['minute_of_day'].clip(lower=0, upper=374)  # Ensure range [0, 374]

    # Normalize "Minute of Day"
    data['minute_of_day'] = data['minute_of_day'] / 374

    data['day_of_week'] = data['date'].dt.dayofweek

    # Drop unused columns
    data = data.drop(columns=['date', 'volume'])

    # Step 2: Normalize other numeric features
    data = normalize_data(data, ['open', 'high', 'low', 'close'])

    # Step 3: Generate labels
    data = generate_labels(data, threshold)

    # Step 4: Create sequences
    X, y = create_sequences(data, sequence_length, target_column='label')

    # Step 5: Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Step 6: Save processed data
    np.save("data/final_data/X_train.npy", X_train)
    np.save("data/final_data/X_test.npy", X_test)
    np.save("data/final_data/y_train.npy", y_train)
    np.save("data/final_data/y_test.npy", y_test)

    # Save for validation
    validation_save(X_train, y_train, "X_train")
    validation_save(X_test, y_test, "X_test")

    print(f"Data preprocessing complete. Saved to 'data/' directory.")
