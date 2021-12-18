import numpy as np
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.corpus import stopwords
import string
from tqdm import tqdm
import fasttext



ft = fasttext.load_model('cc.en.300.bin')
ft.get_dimension()


def is_in_tag(word):
    if word.startswith('<') and word.endswith('>'):
        return True
    return False

def clean_tweets(tweets, text_processor):
    # process words using text processor 
    cleaned = [text_processor.pre_process_doc(s) for s in tqdm(tweets)]
    # remove words with tags <>
    cleaned = [list(filter(lambda word: not is_in_tag(word), tweet)) for tweet in tqdm(cleaned)]
    # remove punctuation
    cleaned =  [list(filter(lambda word: word not in list(string.punctuation), tweet)) for tweet in tqdm(cleaned)]
    # remove stop words 
    stop_words = list(set(stopwords.words('english')))
    cleaned_tweets =  [list(filter(lambda word: word not in stop_words, tweet)) for tweet in tqdm(cleaned)]
    return cleaned_tweets


def get_word_emb(tweet):
    avg_vect = np.zeros(300)
    for word in tweet:
        avg_vect += ft.get_word_vector(word)
    return avg_vect / 300


def pre_process_training_data():
    POS_PATH = "../Twitter_DataSet/train_pos_full_nodup.txt"
    NEG_PATH = "../Twitter_DataSet/train_neg_full_nodup.txt"

    pos_tweets = []
    neg_tweets = []
    with open(POS_PATH,"r") as f:
        pos_tweets = f.readlines()
        f.close()
    with open(NEG_PATH, "r") as f:
        neg_tweets = f.readlines()
        f.close()
    print(len(pos_tweets))
    print(len(neg_tweets))
    
    text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
    )
    
    neg_tweets_cleaned = clean_tweets(neg_tweets)
    pos_tweets_cleaned = clean_tweets(pos_tweets)

    print(len(pos_tweets_cleaned))
    print(len(neg_tweets_cleaned))
    
    # [1, 0] encodes :( tweet
    # [0, 1] encodes :) tweet
    training_data = []
    for tweet in tqdm(neg_tweets_cleaned):
        emb = get_word_emb(tweet)
        training_data.append([np.array(emb), np.array([1.0, 0.0])])

    for tweet in tqdm(pos_tweets_cleaned):
        emb = get_word_emb(tweet)
        training_data.append([np.array(emb),np.array([0.0, 1.0])])

    print("Shuffiling and saving training_data")
    np.random.shuffle(training_data)
    np.save("training_data.npy", training_data)
    
def pre_process_test_dat():
    TEST_PATH = "../Twitter_DataSet/test_data.txt"
    tweets = []
    with open(TEST_PATH,"r") as f:
        tweets = f.readlines()
        f.close()
    text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
    )
    
    cleaned_tweets = clean_tweets(tweets, text_processor)
    test_data = []
    for tweet in tqdm(cleaned_tweets):
        emb = get_word_emb(tweet)
        test_data.append([np.array(emb)])
    print("Shuffiling and saving training_data")
    np.random.shuffle(test_data)
    np.save("test_data.npy", test_data)

    
if __name__ == "__main__": 
    pre_process_test_dat()
        
    