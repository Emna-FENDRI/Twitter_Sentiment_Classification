import numpy as np
import string
import re
from dictionaries import*

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from ekphrasis.classes.segmenter import Segmenter
#from nltk.stem import PorterStemme

seg= Segmenter("twitter")
lm = nltk.WordNetLemmatizer()

def clean_tweet(tweet):
    
    #lower case all words
    tweet = tweet.lower()
    #Normalize to 2 repetition 
    tweet = handle_repeating_char(tweet)
    #Unpack Hashtags
    tweet = unpack_hashtag(tweet)
    # replace any number by "number"
    tweet = replace_numbers(tweet)
    tweet = replace_exclamation(tweet)
    tweet = replace_question(tweet)
    #Translate emoticons
    tweet = handle_emoticons(tweet)
    #remove all punctuation left
    tweet = remove_punct(tweet)
    #remove stop words
    tweet = remove_stop_words(tweet)
    tweet = word_tokenize(tweet)
    tweet = lemmatizer(tweet)
    tweet = ' '.join(word for word in tweet)
    return tweet 

def remove_punct(text):
    text  = ''.join([char for char in text if char not in string.punctuation])
    return text

def replace_numbers(text):
    return re.sub('[0-9]+', ' number ', text)

def replace_exclamation(text):
    return re.sub(r'(\!)+', ' exclamation ', text)

def replace_question(text):
    return re.sub(r'(\?)+', ' question ', text)

def unpack_hashtag(text):
    words = text.split()
    return ' '.join([seg.segment(w[1:]) if (w[0] == '#') else w for w in words ])

def remove_stop_words(text):
    text= text.lower()
    stop_words = set(stopwords.words('english'))
    #word_tokens = word_tokenize(text)
    filtered_sentence = ' '.join([w for w in text.split() if not w in stop_words])
    return filtered_sentence

def handle_repeating_char(text):
    #return re.sub(r'(.)\1+', r'\1', text)
    return re.sub(r'(.)\1+', r'\1\1', text)

def lemmatizer(data):
    return [lm.lemmatize(w) for w in data]  

def handle_emoticons(text):
    text = re.sub('(h*ah+a*h*)+', "<laugh>", text)
    text = re.sub('(h*eh+e*h*)+', "<laugh>", text)
    text= re.sub('(h*ih+i*h*)+', "<laugh>", text)
    return ' '.join(emoticons[w] if w in emoticons else w for w in text.split())
