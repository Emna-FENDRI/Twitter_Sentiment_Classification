import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import *

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

  

