import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import *
from tqdm.notebook import tqdm_notebook


import seaborn as sns
import re
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from helper import *
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import pickle


def get_TFIDF_transformation(X_train, X_test, max_features ,ngram_range):
    TFID_vect = TfidfVectorizer(max_features = max_features ,ngram_range = ngram_range)
    #Transform the data using TF-IDF Vectorizer
    X_train_TFIDF_Matrix = TFID_vect.fit_transform(X_train)
    X_test_TFIDF_Matrix  = TFID_vect.transform(X_test)
    number_of_features = len(TFID_vect.get_feature_names())
    return X_train_TFIDF_Matrix, X_test_TFIDF_Matrix, number_of_features




def naive_bayes_count(X_train, X_test, y_train, y_test, max_features , ngram_range , display_evaluation= False):
    
    count_vect = CountVectorizer(max_features = max_features ,ngram_range = ngram_range)
    #Transform training and test set 
    #Fit on training set => to get feature words with weights
    X_train_count_matrix = count_vect.fit_transform(X_train)
    number_of_features = len(count_vect.get_feature_names())
    X_test_count_matrix = count_vect.transform(X_test)

    MNB_classifier = MultinomialNB().fit(X_train_count_matrix, y_train)
    y_pred = MNB_classifier.predict(X_test_count_matrix)
    
    print("NAIVE BAYES MODEL with CountVectorizer ")
    print("with {} features selected \n".format(number_of_features))
    print("Accuracy:\n",get_accuracy(y_test,y_pred))
    print(classification_report(y_test, y_pred))
    if(display_evaluation):
        evaluate_model(MNB_classifier,  X_test_count_matrix, y_pred,y_test)

    
def naive_bayes_TFIDF(X_train, X_test, y_train, y_test, max_features , ngram_range , display_evaluation= False):
    
    X_train_TFIDF, X_test_TFIDF, number_of_features = get_TFIDF_transformation(X_train, X_test,max_features ,ngram_range)
    
    MNB_classifier = MultinomialNB().fit(X_train_TFIDF, y_train)
    y_pred = MNB_classifier.predict(X_test_TFIDF)
    
    print("NAIVE BAYES MODEL with TFIDF transformation ")
    print("with {} features selected \n ".format(number_of_features))
    print("Accuracy:\n",get_accuracy(y_test,y_pred))
    print(classification_report(y_test, y_pred))
    if(display_evaluation):
        evaluate_model(MNB_classifier,  X_test_TFIDF, y_pred,y_test)

def logistic_regression_TFIDF(X_train, X_test, y_train, y_test,max_features , ngram_range, display_evaluation = False):
    
    X_train_TFIDF, X_test_TFIDF, number_of_features = get_TFIDF_transformation(X_train, X_test,max_features,ngram_range)
    clf = LogisticRegression().fit( X_train_TFIDF, y_train)
    y_pred = clf.predict( X_test_TFIDF)
    
    print("LOGISTIC REGRESSION MODEL with TFIDF transformation ")
    print("with {} features selected \n ".format(number_of_features))
    print("Accuracy:",get_accuracy(y_test,y_pred))
    print(classification_report(y_test, y_pred))
    if (display_evaluation):
        evaluate_model(clf,  X_test_TFIDF, y_pred,y_test)

def SVM_TFIDF(X_train, X_test, y_train, y_test, max_features, ngram_range , display_evaluation = False):
    
    X_train_TFIDF, X_test_TFIDF, number_of_features = get_TFIDF_transformation(X_train, X_test,max_features,ngram_range)
    clf = svm.SVC()
    clf.fit( X_train_TFIDF, y_train)
    y_pred = clf.predict( X_test_TFIDF)
    
    print("SVM MODEL with TFIDF transformation ")
    print("with {} features selected \n ".format(number_of_features))
    print("Accuracy:\n",get_accuracy(y_test,y_pred))
    print(classification_report(y_test, y_pred))
    if (display_evaluation):
        evaluate_model(clf,  X_test_TFIDF, y_pred,y_test)


def Tweet_to_GloVe(tweet,embeddings_index):
    tweet_list = tweet.split()
    #tweet_list_filtered =  list(set(tweet_list) & set(embeddings_index.keys()))  #remove words not in embedding 
    list_of_embeddings = np.array([embeddings_index.get(word, np.zeros(25)) for word in tweet_list],dtype='float64')
    x = np.mean(list_of_embeddings, axis = 0)
    #x[np.isinf(x)] = 0 #sanitize infinity
    return np.nan_to_num(x)

def get_Glove_transformation(tweets, dimension, preprocessed):
    if preprocessed:
        with open('../Data/preprocessed_embeddings_index.pkl', 'rb') as f:
            embeddings_index = pickle.load(f)
    else:
        with open('../Data/embeddings_index.pkl', 'rb') as f:
            embeddings_index = pickle.load(f)     

    x = np.zeros((len(tweets), dimension))

    for i in tqdm_notebook(range(len(tweets))):
        x[i] = Tweet_to_GloVe(tweets[i],embeddings_index)
    return x
def SVM_Glove(X_train, X_test, y_train, y_test, number_of_features, display_evaluation = False, preprocessed = False):
    
    X_train_glove = get_Glove_transformation(X_train, number_of_features, preprocessed)
    X_test_glove = get_Glove_transformation(X_test, number_of_features, preprocessed)
    
    with open('../Data/X_train_glove.pkl', 'wb') as f:
        pickle.dump(X_train_glove, f)
    with open('../Data/X_test_glove.pkl', 'wb') as f:
        pickle.dump(X_test_glove, f)
    print("finished embedding")
    clf = svm.SVC()
    clf.fit( X_train_glove, y_train)
    y_pred = clf.predict(X_test_glove)
    
    print("SVM MODEL with TFIDF transformation ")
    print("with {} features selected \n ".format(number_of_features))
    print("Accuracy:\n",get_accuracy(y_test,y_pred))
    print(classification_report(y_test, y_pred))
    if (display_evaluation):
        evaluate_model(clf,  X_test_glove, y_pred,y_test)
  

