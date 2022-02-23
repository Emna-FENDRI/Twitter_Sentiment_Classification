import numpy as np
import os
import pandas as pd
import numpy as np 
import pickle
import csv
from time import time
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Masking, Embedding, Bidirectional, LSTM,Dense, Activation, GRU


from google_drive_downloader import GoogleDriveDownloader as gdd


if(not os.path.exists('clean_pos_train_with_stopWords.txt')):
    gdd.download_file_from_google_drive(file_id='1CZ1enlwtQZ61LwhzjucAAD4i-wf1rUT4',
                                    dest_path='./clean_pos_train_with_stopWords.txt',
                                    unzip=True)
if(not os.path.exists('clean_neg_train_with_stopWords.txt')):
    gdd.download_file_from_google_drive(file_id='15QED31vdUzCTKN2lm5WWgmwqLLJZbhGC',
                                    dest_path='./clean_neg_train_with_stopWords.txt',
                                    unzip=True) 
if(not os.path.exists('clean_test_with_stopWords.txt')):
    gdd.download_file_from_google_drive(file_id='1WOv_VBUu3ZNzZSNGHY5mrBTgfy8zyJ6z',
                                    dest_path='./clean_test_with_stopWords.txt',
                                    unzip=True)
if(not os.path.exists('multi_cnn')):
    gdd.download_file_from_google_drive(file_id='1p07OkoJ693qLvJrEXIxQFTfee57cKwM8',
                                    dest_path='./multi_cnn.zip',
                                    unzip=True)
if(not os.path.exists('glove.6B.200d.txt')):
    gdd.download_file_from_google_drive(file_id='1JoWTlKlibfKV-eK0J7DOSlGeDRHt5S9E',
                                    dest_path='./glove.6B.200d.txt',
                                    unzip=True)
                                        
def data_tokenizer(X_train,X_test, maxlen = 100,num_words = 50000):

    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(X_train)

    X_train_tokenized = tokenizer.texts_to_sequences(X_train)
    X_test_tokenized = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1
    X_train_tokenized_pad = pad_sequences(X_train_tokenized, padding='post', maxlen=maxlen)
    X_test_tokenized_pad = pad_sequences(X_test_tokenized, padding='post', maxlen=maxlen)
    return X_train_tokenized_pad,X_test_tokenized_pad, tokenizer.word_index , vocab_size 

def create_embedding_matrix(word_index, embedding_dim = 200,filepath = 'glove.6B.200d.txt', custom = False):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

        
    if custom:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
    else:
        with open(filepath,encoding = "utf8") as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def main():
    #load clean data 
    with open('clean_pos_train_with_stopWords.txt',"r",encoding = "utf8") as file:
        pos = file.read().split('\n')
    pos = pd.DataFrame({'tweet' : pos})[:len(pos)-1]

    with open('clean_neg_train_with_stopWords.txt',"r",encoding = "utf8") as file:
        neg = file.read().split('\n')
    neg = pd.DataFrame({'tweet' : neg})[:len(neg)-1]

    pos['target'] = 1
    neg['target'] = 0

    #concat pos and neg to form full training set
    train= pd.concat([pos, neg])
    X_train = train.tweet
    y_train = train.target

    #####  FOR SUBMISSION: 
    #load cleaned test set

    with open('clean_test_with_stopWords.txt',"r") as file:
        X_test = file.read().split('\n')
    X_test = pd.DataFrame({'tweet' : X_test})[:len(X_test)-1]

    #Return data tokenized, with index mapping to words
    X_train_tok,X_test_tok,word_dict , vocab_size = data_tokenizer(X_train,X_test.tweet)
    #get embedding matrix (GloVe 200)
    embed_matrix = create_embedding_matrix(word_dict)

    #load neural net model
    bi_dir_model = keras.models.load_model("multi_cnn_glove_trained")

    # get final prediction
    y_pred = bi_dir_model.predict(X_test_tok)
    y_pred[y_pred >= 0.5 ] = 1
    y_pred[y_pred < 0.5 ] = -1


    ids = np.arange(1, len(y_pred)+1)


    with open('./submission.csv', 'w',newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

if __name__ == "__main__":
    main()
