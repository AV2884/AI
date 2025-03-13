import nltk 
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string
import numpy as np
import pandas as pd

#STEP 1: (data prep)----------------------------------------------------------------------

# Download datasets
nltk.download('twitter_samples')
nltk.download('stopwords')

# Load dataset
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Initialize tokenizer and stemmer
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def process_tweet(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)              # Remove RT(old type of tweets)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)   # Remove links
    tweet = re.sub(r'#', '', tweet)                     # Remove # symbol

    tokens = tokenizer.tokenize(tweet)                  #Every string after a space
    
    # Remove stopwords & punctuation, apply stemming
    clean_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return clean_tokens


#STEP 2: (Frequency Table)------------------------------------------------------------------

def build_freq_table():
    pass

