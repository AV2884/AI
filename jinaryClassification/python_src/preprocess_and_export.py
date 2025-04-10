# File: python_src/preprocess_and_export.py

import numpy as np
import os
from tensorflow.keras.datasets import mnist


def prepare_data():
    print(f"💽 Date prep")
    # Prepare data folder
    data_dir = "/root/aiRoot/0-AI/AI/jinaryClassification/data"
    os.makedirs(data_dir, exist_ok=True)

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Filter only digits 0 and 1
    train_mask = (y_train == 0) | (y_train == 1)
    test_mask = (y_test == 0) | (y_test == 1)

    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask]

    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]

    # Normalize and flatten
    x_train_filtered = x_train_filtered / 255.0
    x_train_filtered = x_train_filtered.reshape(x_train_filtered.shape[0], -1)
    y_train_filtered = y_train_filtered.reshape(-1, 1)

    x_test_filtered = x_test_filtered / 255.0
    x_test_filtered = x_test_filtered.reshape(x_test_filtered.shape[0], -1)
    y_test_filtered = y_test_filtered.reshape(-1, 1)

    # Save to CSV
    np.savetxt(os.path.join(data_dir, "x_train_filtered.csv"), x_train_filtered, delimiter=",")
    np.savetxt(os.path.join(data_dir, "y_train_filtered.csv"), y_train_filtered, delimiter=",")
    np.savetxt(os.path.join(data_dir, "x_test_filtered.csv"), x_test_filtered, delimiter=",")
    np.savetxt(os.path.join(data_dir, "y_test_filtered.csv"), y_test_filtered, delimiter=",")

    print(f"💽 [✓] Saved TRAIN set: {x_train_filtered.shape}, {y_train_filtered.shape}")
    print(f"💽 [✓] Saved TEST set : {x_test_filtered.shape}, {y_test_filtered.shape}")
