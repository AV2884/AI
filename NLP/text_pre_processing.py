import nltk
from nltk.corpus import twitter_samples
import re                                  
import string                              
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
from collections import Counter

nltk.download('twitter_samples')
nltk.download('stopwords')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

tweet1 = all_positive_tweets[0]

print("ORIGINAL TWEET","-"*100)
print(f"\n{tweet1}\n")  

def print_removed_part(string1, string2):
    count1 = Counter(string1)
    count2 = Counter(string2)
    removed_chars = []
    for char in count1:
        if count1[char] > count2.get(char, 0):  # Check if char is missing or reduced
            removed_chars.extend([char] * (count1[char] - count2.get(char, 0)))
    print("Removed Characters:", removed_chars)
    return removed_chars


#Remove hyperlinks, Twitter marks and styles
print("="*100)
tweet2 = re.sub(r'^RT[\s]+', '', tweet1)           # remove old style retweet text "RT"
tweet2 = re.sub(r'https?://[^\s\n\r]+', '', tweet2)# remove hyperlinks
tweet2 = re.sub(r'#', '', tweet2)                  # only removing the hash # sign from the word
print("Step 1: Remove hyperlinks, Twitter marks and styles \n")
print_removed_part(tweet1, tweet2)
print(tweet2,"\n")
print("="*100)

#Tokenize the string
print("="*100)
print("Step 2: Tokenize the tweet: \n")
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet2)
print('Tokenized string:')
print(tweet_tokens)
print("="*100)
print()

#Remove stop words and punctuations
print("="*100)
print("Step 3: Remove stop words and punctuations:")
stopwords_english = stopwords.words('english') 
print('Stop words:')
print(stopwords_english)
print('Punctuation:')
print(string.punctuation)
print("="*100)

tweets_clean = []
for word in tweet_tokens:
    if (word not in stopwords_english and word not in string.punctuation):
        tweets_clean.append(word)
print(tweets_clean)
print("="*100)

#Stemming:
print("="*100)
print("Step 4: Stemming:")
stemmer = PorterStemmer() 
tweets_stem = []
for word in tweets_clean:
    stem_word = stemmer.stem(word)  # stemming word
    tweets_stem.append(stem_word)  # append to the list
print('stemmed words:')
print(tweets_stem)
print("="*100)


print("INIT TWEET")
print(tweet1)
print("FINAL TWEET")
print(tweets_stem)


#---------------ONE SHOT PROCCESS---------------------
# choose the same tweet
# tweet = all_positive_tweets[2277]

# # call the imported function
# tweets_stem = process_tweet(tweet); # Preprocess a given tweet

# print('preprocessed tweet:')
# print(tweets_stem) # Print the result