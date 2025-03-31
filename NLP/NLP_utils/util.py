import re                                  
import string                              
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
import numpy as np

def process_tweet(tweet):
    # Step 1: Remove Twitter styles and unwanted text
    tweet = re.sub(r'^RT[\s]+', '', tweet)  # Remove RT
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)  # Remove hyperlinks
    tweet = re.sub(r'#', '', tweet)  # Remove only the # symbol

    # Step 2: Tokenize the tweet
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    # Step 3: Remove stopwords and punctuation
    stopwords_english = set(stopwords.words('english'))  # Use set for faster lookup
    tweets_clean = [word for word in tweet_tokens if word not in stopwords_english and word not in string.punctuation]

    # Step 4: Apply Stemming
    stemmer = PorterStemmer()
    tweets_stem = [stemmer.stem(word) for word in tweets_clean]

    return tweets_stem  # Return processed tweet


def build_freqs(tweets, ys):
    freqs = {}
    yslist = np.squeeze(ys).tolist() 

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet): 
            pair = (word, y)  
            freqs[pair] = freqs.get(pair, 0) + 1 
    return freqs













def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    ### START CODE HERE ###
    # use lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = (lookup(freqs,word,1) )
    
    # use lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = lookup(freqs,word,0)
    
    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = pos_neg_ratio['positive'] / pos_neg_ratio['negative']
    ### END CODE HERE ###
    return pos_neg_ratio