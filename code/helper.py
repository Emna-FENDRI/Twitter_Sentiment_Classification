import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix




PATH_TRAIN_POS = '../Twitter_DataSet/train_pos.txt'
PATH_TRAIN_NEG = '../Twitter_DataSet/train_neg.txt'


def load_data(path_train_pos = PATH_TRAIN_POS , path_train_neg = PATH_TRAIN_NEG ):
    pos_train = pd.read_csv(PATH_TRAIN_POS, sep = '\r',  names = ['Text'])
    pos_train.insert(1, 'Target', 1)
    neg_train = pd.read_csv(PATH_TRAIN_NEG, sep = '\r',  names = ['Text'])
    neg_train.insert(1, 'Target', -1)
    return pos_train , neg_train


def split_data(pos_train , neg_train):
    #Merge Pos and Neg => Create Train_set
    train_set= pd.concat([pos_train, neg_train])
    X=  train_set.Text
    y= train_set.Target
    #SPLIT: Set same random_state to reproduce same result
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=13)

    print("Train_set Info: SIZE= {size}, POSITIVE Tweets ={pos:0.2f}%, NEGATIVE Tweets = {neg:0.2f}%".format( size= len(X_train),
                                                                           pos = len(y_train[y_train == 1])*100/len(X_train),
                                                                           neg = len(y_train[y_train == -1])*100/len(X_train)))

    print("Test_set Info: SIZE= {size}, POSITIVE Tweets ={pos:0.2f}%, NEGATIVE Tweets = {neg:0.2f}%".format( size= len(X_test),
                                                                           pos = len(y_test[y_test == 1])*100/len(X_test),
                                                                           neg = len(y_test[y_test == -1])*100/len(X_test)))
    return X_train, X_test, y_train, y_test

def get_accuracy(y_test, y_pred):
    return round(accuracy_score(y_test, y_pred), 2)


def evaluate_model(model,X_test ,y_pred, y_test) :
    

    
    #Show Confusion Matrix
    c_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in c_matrix.flatten() / np.sum(c_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(c_matrix, annot = labels, fmt = '',  xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show() 
    
    
    ##Show ROC
    # generate a no skill prediction 
    ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    model_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    model_probs = model_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    model_auc = roc_auc_score(y_test, model_probs)
    # summarize scores

    #print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Model: ROC AUC=%.3f' % (model_auc))
    
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(model_fpr, model_tpr, marker='.', label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()    