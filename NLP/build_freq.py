import nltk                                  # Python library for NLP
from nltk.corpus import twitter_samples      # sample Twitter dataset from NLTK
import numpy as np                           # library for scientific computing and matrix operations
from NLP_utils.util import process_tweet
nltk.download('twitter_samples')
nltk.download('stopwords')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = all_positive_tweets + all_negative_tweets


print("Number of tweets: ", len(tweets))

labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))
print(labels)

def build_freqs(tweets, ys):
    """
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its frequency
        [word, sentiment, frequency]
    """
    freqs = {}
    yslist = np.squeeze(ys).tolist() 

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet): 
            pair = (word, y)  
            freqs[pair] = freqs.get(pair, 0) + 1 
    return freqs

freqs = build_freqs(tweets,labels)
# create frequency dictionary
freqs = build_freqs(tweets, labels)

# check data type
print(f'type(freqs) = {type(freqs)}')

# check length of the dictionary
print(f'len(freqs) = {len(freqs)}')
# print(freqs)