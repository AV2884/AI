from sklearn.model_selection import train_test_split
import time
import nltk
import re
import pickle 
import string
import numpy as np
import pandas as pd
import os
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm


# ============================
# 🚀 HYPERPARAMETERS
# ============================
ALPHA = 0.001        # Learning rate
EPOCHS = 1000       # Number of training iterations
CHECKPOINT_EVERY = 200  # Save model every X epochs
CHECKPOINT_DIR = "checkpoints"
FREQ_TABLE_PATH = os.path.join(CHECKPOINT_DIR, "freq_table.pkl")
LOAD_TRAINED_MODEL = True

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================
# 🔹 STEP 1: DATA PREPARATION
# ============================
print("[INFO] Downloading Twitter dataset...")
nltk.download('twitter_samples')
nltk.download('stopwords')

# Load dataset
print("[INFO] Loading tweets...")
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


print(f"[INFO] Loaded {len(all_positive_tweets)} positive and {len(all_negative_tweets)} negative tweets.")

# Initialize tokenizer and stemmer
print("[INFO] Initializing tokenizer and stemmer...")
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ============================
# 🔹 STEP 2: PREPROCESSING FUNCTION
# ============================
def preprocess_tweet(tweet):
    """
    Preprocesses a tweet by removing links, mentions, hashtags,
    tokenizing, removing stopwords, and applying stemming.
    """
    tweet = re.sub(r'^RT[\s]+', '', tweet)              # Remove RT (old retweet format)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)   # Remove links
    tweet = re.sub(r'#', '', tweet)                     # Remove # symbol

    tokens = tokenizer.tokenize(tweet)  # Tokenize words

    # Remove stopwords & punctuation, apply stemming
    clean_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return clean_tokens

# ============================
# 🔹 STEP 3: BUILD FREQUENCY TABLE
# ============================
def build_freq_table(tweets, labels):
    """
    Builds a frequency table storing occurrences of words in positive and negative tweets.
    """
    freq_table = {}

    for tweet, label in zip(tweets, labels):
        for word in tweet:
            if word not in freq_table:
                freq_table[word] = [0, 0]  # Initialize word count
            if label == 1:
                freq_table[word][1] += 1  # Positive count
            else:
                freq_table[word][0] += 1  # Negative count

                
    with open(FREQ_TABLE_PATH, "wb") as f:
        pickle.dump(freq_table, f)

    return freq_table    

# ============================
# 🔹 STEP 4: PREPROCESS TWEETS
# ============================
print("[INFO] Preprocessing tweets...")
processed_positive_tweets = [preprocess_tweet(tweet) for tweet in all_positive_tweets]
processed_negative_tweets = [preprocess_tweet(tweet) for tweet in all_negative_tweets]
print(f"[INFO] Preprocessed {len(processed_positive_tweets)} positive and {len(processed_negative_tweets)} negative tweets.")

# Merge dataset with labels
tweets = processed_positive_tweets + processed_negative_tweets
labels = np.concatenate((np.ones(len(all_positive_tweets)), np.zeros(len(all_negative_tweets))))

# ============================
# 🔹 STEP 5: BUILD FREQUENCY TABLE
# ============================
print("[INFO] Building frequency table...")
freq_table = build_freq_table(tweets, labels)

# Print a sample from the frequency table
print("\n[INFO] Sample word frequencies:")
print("Word ---> Neg f , Pos f")
for i, (key, value) in enumerate(freq_table.items()):
    print(f"{key} --> {value}")
    if i > 5: break  # Print only first 6 words

# ============================
# 🔹 STEP 6: LOGISTIC REGRESSION
# ============================


def load_latest_checkpoint():
    """
    Loads the latest available model checkpoint.
    Returns the weights and the epoch number from which to resume.
    """
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("weights_epoch")]

    if not checkpoint_files:
        tqdm.write("[INFO] No saved model found. Starting training from scratch.")
        return None, 0  # No checkpoint found, start from scratch

    # Sort by epoch number (extracting numbers from filenames)
    checkpoint_files.sort(key=lambda f: int(f.split("_")[2].split(".")[0]))

    latest_checkpoint = checkpoint_files[-1]  # Get the latest file
    epoch_number = int(latest_checkpoint.split("_")[2].split(".")[0])  # Extract epoch number

    weights_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
    weights = np.load(weights_path)  # Load weights

    tqdm.write(f"[INFO] Resuming training from checkpoint: {latest_checkpoint} (Epoch {epoch_number})")
    return weights, epoch_number


def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))

def extract_features(tweet, freq_table):
    x = np.zeros(1 + len(freq_table)) 

    x[0] = 1  # Bias term

    for word in tweet:
        if word in freq_table:
            x[1 + list(freq_table.keys()).index(word)] = sum(freq_table[word])  # Total frequency
    
    return x

def initialize_weights(n):
    return np.zeros(n)

def compute_loss(y, y_hat):
    epsilon = 1e-9
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)

def save_checkpoint(weights, epoch):
    filename = os.path.join(CHECKPOINT_DIR, f"weights_epoch_{epoch}.npy")
    np.save(filename, weights)

def gradient_descent(X, y, w, alpha, epochs):
    """
    Perform batch gradient descent with structured progress reporting.

    Args:
    - X: Feature matrix
    - y: Labels
    - w: Weights
    - alpha: Learning rate
    - epochs: Number of iterations

    Returns:
    - w: Updated weights
    - loss_history: Loss at each step
    """
    print(f"Feature Vector Shape: {X.shape}")
    print(f"Feature Vector Sample: {X[:5]}")

    m = len(y)
    loss_history = []
    start_time = time.time()  # Track total training time
    previous_loss = None  # To calculate delta loss

    # Print training start message
    tqdm.write(f"\n[INFO] Starting training with {epochs} epochs...\n")

    # Overall tqdm progress bar
    pbar = tqdm(total=epochs, desc="[INFO] Training Progress", unit="epoch", position=1, leave=True)


    for i in range(epochs):
        iter_start_time = time.time()  # Track time per iteration

        # Compute forward pass
        z = np.dot(X, w)
        y_hat = sigmoid(z)

        # Compute loss
        loss = compute_loss(y, y_hat)
        loss_history.append(loss)

        # Compute gradient & update weights
        grad = np.dot(X.T, (y_hat - y)) / m
        w -= alpha * grad

        # Calculate elapsed time and estimated remaining time
        total_elapsed_time = time.time() - start_time
        remaining_iters = epochs - (i + 1)  # Remaining iterations
        avg_time_per_iter = total_elapsed_time / (i + 1)
        estimated_completion_time = remaining_iters * avg_time_per_iter

        # Compute delta loss
        delta_loss = previous_loss - loss if previous_loss is not None else None

        # Print loss updates every 10 epochs
        if i % 10 == 0:
            accuracy = compute_accuracy(X, y, w)
            if delta_loss is not None:
                tqdm.write(
                    f"Epoch <{i:5d}> : Loss {loss:.7f} : ΔLoss {delta_loss: .7f} | "
                    f"ETC: {estimated_completion_time:.2f}s | T: {total_elapsed_time:.2f} s | accuracy = {accuracy:.2f}%"
                )
            else:
                tqdm.write(f"Epoch <{i}> : Loss {loss:.7f} : ΔLoss N/A | T: {total_elapsed_time:.2f} s | accuracy = {accuracy:.2f}%")

            

            previous_loss = loss  # Update previous loss for next iteration

        # Update tqdm progress bar
        pbar.update(1)

        # Save weights at checkpoint intervals
        if i % CHECKPOINT_EVERY == 0 and i != 0:
            save_checkpoint(w, i)

    pbar.close()  # Close tqdm progress bar after training

    # Final training message
    end_time = time.time()
    training_time = end_time - start_time
    tqdm.write(f"[INFO] Training completed in {training_time:.2f} seconds.")

    # Save final weights after training
    final_checkpoint = os.path.join(CHECKPOINT_DIR, "final_weights.npy")
    np.save(final_checkpoint, w)
    tqdm.write("[INFO] Final weights saved successfully.")

    return w, loss_history


def compute_accuracy(X, y, w):
    z = np.dot(X, w)  # Compute model output
    y_hat = sigmoid(z)  # Convert to probability
    predictions = (y_hat >= 0.5).astype(int)  # Convert probabilities to 0 or 1
    print(f"Unique Predictions: {np.unique(predictions, return_counts=True)}")
    correct_predictions = np.sum(predictions == y)
    accuracy = (correct_predictions / len(y)) * 100

    return accuracy

#=-=-=-=-=-=-==-=-=-=-=-=-==-=-=-=-=-=-==-=-=-=-=-=-==-=-=-=-=-=-==-=-=-=-=-=-==-=-=-=-=-=-==-=-=-=-=-=-==-=-=-=-=-=-=

print("[INFO] Extracting features...")
X = np.array([extract_features(tweet, freq_table) for tweet in tweets])

# Normalize features to prevent large values
X = X / (np.max(X, axis=0) + 1e-8)  # Adding small value to avoid division by zero
y = np.array(labels)

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[INFO] Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


print("[INFO] Initializing weights...")
w = initialize_weights(X_train.shape[1])  # Initialize based on training features

start_epoch = 0  # Default starting epoch

if LOAD_TRAINED_MODEL:
    w, start_epoch = load_latest_checkpoint()
    if w is None:  # If no checkpoint exists, start fresh
        print("[INFO] No saved model found. Initializing fresh weights.")
        w = initialize_weights(X_train.shape[1])
        start_epoch = 0
else:
    print("[INFO] Training from scratch...")
    w = initialize_weights(X_train.shape[1])

print(f"[INFO] Starting training from epoch {start_epoch} to {EPOCHS}...")
w_before = w.copy()  # Store initial weights

# Train only on the training set
w, loss_history = gradient_descent(X_train, y_train, w, ALPHA, EPOCHS)

w_after = w  # Store updated weights

print("[INFO] Training complete! Saving final model...")
final_checkpoint = os.path.join(CHECKPOINT_DIR, "final_weights.npy")
np.save(final_checkpoint, w)
print(f"[INFO] Final weights saved at {final_checkpoint}")

train_accuracy = compute_accuracy(X_train, y_train, w)
test_accuracy = compute_accuracy(X_test, y_test, w)

print(f"[INFO] Weight Change During Training: {np.sum(np.abs(w_after - w_before)):.6f}")
print(f"[INFO] Train Accuracy: {train_accuracy:.2f}%")
print(f"[INFO] Test Accuracy: {test_accuracy:.2f}%")