import numpy as np
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.corpus import stopwords
import string
from tqdm import tqdm
import fasttext
import fasttext.util
from nltk.tokenize import word_tokenize

WORD_VECTOR_DIM = 100

def remove_stop_words(tweets):
    '''Removes stop words from tweets'''
    stop_words = list(set(stopwords.words('english')))
    cleaned_tweets =[]
    for tweet in tqdm(tweets):
        cleaned_tweet = []
        for word in word_tokenize(tweet):
            if word not in stop_words:
                cleaned_tweet.append(word)
        cleaned_tweets.append(cleaned_tweet)
    return cleaned_tweets


def get_word_emb(tweet, ft):
    '''Generates a word embedding for tweet by adding the word embeddings of each word'''
    avg_vect = np.zeros(WORD_VECTOR_DIM)
    for word in tweet:
        emb = ft.get_word_vector(word)
        avg_vect += emb
    return avg_vect

def get_word_embeddings(ft, out_name):
    '''Creates and saves the traning set using passed fasttext model'''
    POS_PATH = "clean_pos_train_with_stopWords.txt"
    NEG_PATH = "clean_neg_train_with_stopWords.txt"

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
    
    training_data_avg = []
    pos_tweets = remove_stop_words(pos_tweets)
    neg_tweets = remove_stop_words(neg_tweets)

    for tweet in tqdm(neg_tweets):
        emb = get_word_emb(tweet, ft)
        training_data_avg.append([np.array(emb), np.array([1.0, 0.0])])

    for tweet in tqdm(pos_tweets):
        emb = get_word_emb(tweet, ft)
        training_data_avg.append([np.array(emb), np.array([0.0, 1.0])])

    print("Shuffiling and saving training_data")
    np.random.shuffle(training_data_avg)

    np.save(f"{out_name}.npy", training_data_avg)
    print("Saved training data")

    
if __name__ == "__main__": 
    ft = fasttext.load_model('custom_fasttext.bin')
    # reduce dim to 100
    fasttext.util.reduce_model(ft, 100)
    print(f"Loaded model, word embeddings have dim = {ft.get_dimension()}")
    get_word_embeddings(ft, "fastext_training")
    pre_process_test_dat(ft)
    
        
    
