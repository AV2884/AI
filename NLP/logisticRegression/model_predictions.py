import os
import numpy as np
import pickle
import re
import string
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import twitter_samples

# ============================
# 🚀 CONFIGURATION
# ============================
CHECKPOINT_DIR = "checkpoints"
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final_weights.npy")
FREQ_TABLE_PATH = os.path.join(CHECKPOINT_DIR, "freq_table.pkl")

# Load NLP tools
nltk.download('stopwords')
nltk.download('twitter_samples')
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ============================
# 🔹 LOAD TRAINED MODEL
# ============================
def load_model():
    if os.path.exists(FINAL_MODEL_PATH):
        return np.load(FINAL_MODEL_PATH)
    else:
        tqdm.write("[ERROR] No trained model found. Train the model first.")
        exit()

# ============================
# 🔹 PREPROCESS TWEET
# ============================
def preprocess_tweet(tweet):
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)  # Remove links
    tokens = tokenizer.tokenize(tweet)
    return [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]

# ============================
# 🔹 BUILD FREQUENCY TABLE
# ============================


# Load and preprocess dataset
print("[INFO] Loading Twitter dataset...")
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

processed_positive_tweets = [preprocess_tweet(tweet) for tweet in all_positive_tweets]
processed_negative_tweets = [preprocess_tweet(tweet) for tweet in all_negative_tweets]

tweets = processed_positive_tweets + processed_negative_tweets
labels = np.concatenate((np.ones(len(all_positive_tweets)), np.zeros(len(all_negative_tweets))))

# Create new frequency table
print("[INFO] Building new frequency table...")

# Load the saved frequency table
if os.path.exists(FREQ_TABLE_PATH):
    with open(FREQ_TABLE_PATH, "rb") as f:
        freq_table = pickle.load(f)
    print("[INFO] Loaded frequency table from training.")
else:
    print("[ERROR] No frequency table found! Train the model first.")
    exit()


# ============================
# 🔹 EXTRACT FEATURES
# ============================
def extract_features(tweet):
    """
    Converts a preprocessed tweet into a feature vector using the trained frequency table.
    """
    x = np.zeros(1 + len(freq_table))  # +1 for bias
    x[0] = 1  # Bias term
    for word in tweet:
        if word in freq_table:
            x[1 + list(freq_table.keys()).index(word)] = sum(freq_table[word])
    return x

# ============================
# 🔹 MAKE PREDICTION
# ============================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(tweet, model_weights):
    processed_tweet = preprocess_tweet(tweet)
    features = extract_features(processed_tweet)
    prediction = sigmoid(np.dot(features, model_weights))
    return "Positive 😊" if prediction >= 0.5 else "Negative 😞"


# ============================
# 🔹 RUN PREDICTION
# ============================
if __name__ == "__main__":
    print("[INFO] Loading trained model...")
    model_weights = load_model()

    while(True):
        tweet_input = input("\nEnter a tweet to analyze sentiment: ")
        sentiment = predict(tweet_input, model_weights)
    
        print(f"\nPredicted Sentiment: {sentiment}")
