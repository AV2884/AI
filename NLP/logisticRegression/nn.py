import nltk 
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
import re
import string
import numpy as np
import pandas as pd

#STEP 1: (data prep)----------------------------------------------------------------------

# Download datasets
nltk.download('twitter_samples')
nltk.download('stopwords')
print("Data set downloaded")

# Load dataset
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Initialize tokenizer and stemmer
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
print("Tokenizer initialised")


def preprocess_tweet(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)              # Remove RT(old type of tweets)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)   # Remove links
    tweet = re.sub(r'#', '', tweet)                     # Remove # symbol

    tokens = tokenizer.tokenize(tweet)                  #Every string after a space
    
    # Remove stopwords & punctuation, apply stemming
    clean_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return clean_tokens


#STEP 2: (Frequency Table)------------------------------------------------------------------

def build_freq_table(tweets, labels):
    freq_table = defaultdict(lambda: [0, 0])
    
    for tweet, label in zip(tweets, labels):
        for word in tweet:
            if label == 1:
                freq_table[word][1] += 1  # Positive count
            else:
                freq_table[word][0] += 1  # negitive count

    return freq_table    
print("Processign tweets")
processed_positive_tweets = [preprocess_tweet(tweet) for tweet in all_positive_tweets]
processed_negative_tweets = [preprocess_tweet(tweet) for tweet in all_negative_tweets]
print("Processed tweets:")
print(processed_positive_tweets[0:4])

# Merge dataset with labels
tweets = processed_positive_tweets + processed_negative_tweets
labels = np.concatenate((np.ones(len(all_positive_tweets)), np.zeros(len(all_negative_tweets))))

print("Building frequency table")
freq_table = build_freq_table(tweets, labels)
print("Word ---> Neg f , Pos f")
print_counter = 0
for key , value in freq_table.items():
    print(f"{key} --> {value}")
    print_counter += 1
    if print_counter > 5:
        break


#STEP 3 (logistic regression)-----------------------------------------------------


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


def gradient_descent(X, y, w, alpha, epochs):
    """
    Perform batch gradient descent to optimize weights.

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
    m = len(y)
    loss_history = []

    for i in range(epochs):
        z = np.dot(X, w)
        y_hat = sigmoid(z)

        loss = compute_loss(y, y_hat)
        loss_history.append(loss)

        grad = np.dot(X.T, (y_hat - y)) / m
        w -= alpha * grad

        if i % 100 == 0:
            print(f"Epoch {i}: Loss = {loss}")

    return w, loss_history

# Convert all tweets to feature vectors
X = np.array([extract_features(tweet, freq_table) for tweet in tweets])
X = X / np.max(X, axis=0)  # Prevent large values
y = np.array(labels)

# Initialize weights
w = initialize_weights(X.shape[1])

# Train model
alpha = 100
epochs = 100000
w, loss_history = gradient_descent(X, y, w, alpha, epochs)

