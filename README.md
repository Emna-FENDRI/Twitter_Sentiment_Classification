# EPFL CS-433 Machine Learning Project: Sentiment Analysis

## Team (AAE) Members:
Emna Fendri : emna.fendri@epfl.ch

Ahmed Ezzo : ahmed.ezzo@epfl.ch

Mohamed ELasfoury : mohamed.elasfoury@epfl.ch


## Introduction :
The goal of our project is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text. We discuss in our paper various models for this task and explain in more details the steps we followed including the cleaning and the results.

The data set we used to train and test our model can be downloaded from the following AIcrowd page https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files and consists of the following files : 

* ```train_pos_full.txt```: A set of about 1M positive tweets (i.e. happy smiley removed).

* ```train_neg_full.txt```: A set of about 1M negative tweets (i.e. sad smiley removed).

* ```test_data.txt```: The test set, that is the 10000 tweets for which we predict the sentiment label.

## Dependencies :
To be able to run our code, you will need the following libraries.
* ```re```
* ```pandas```
* ```numpy```
* ```scipy```
* ```pickle```
* ```scikit-learn```
* ```keras``` with backend ```tensorflow 2.7.0``` installed and configured
* ```nltk``` for preprocessing, make sure to also download the dicts we used for cleaning. For this, you can run these lines.
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
* ```ekphrasis``` that we used for cleaning (in particular for spell corrector)
```
!pip install ekphrasis
```
* ```google_drive_downloader```

## Files :

* ```BaseLine_models.py```: Implementation of basic machine learning models, made for local tests.
* ```nn_models.py``` : Implementation of our deep learning models.
* ```cleaning.py```: All the methods used for our cleaning process.
* ```helper.py```: useful functions.
* ```Voting_algorithm.ipynb```: Notebook used to make a final prediction based on 3 of our models (Ensemble learning).
* ```run.py```: To reproduce our best model (Bidir-LSTM).
* ```preprocess_fasttext.py```: To generate the training and test data for the FNN using cbow word embeddings.
* ```Basic_NN.ipynb```: Notebook that runs the FNN on the Cbow word embeddings.

There exists some files that are too big to upload on github, you can find them in [this google drive folder](https://drive.google.com/drive/folders/1UXxhAXK1MTMsKBNRyCM0CLKD_mLsskm4?usp=sharing)
This is a breakdown of the content of the files inside:
* ```clean_neg_train_with_stopWords.txt```: Cleaned negative tweets from the training set
* ```clean_pos_train_with_stopWords.txt```: Cleaned positive tweets from the training set
* ```clean_test_with_stopWords.txt```: Cleaned test set
* ```BidirLSTM.zip```: The neural network model used to get the best submission
* ```cooc.pkl```: The coocurence matrix of the cleaned training set
* ```vocab_cut.txt```: The vocabulary of tweets of the training set
* ```glove.6B.200d.txt```: Preprocessed GloVe embedding found [from GloVe's creators's Github](https://github.com/stanfordnlp/GloVe) 

### Custom_GloVe file :

You can download ```vocab_cut.txt``` and  ```cooc.pkl``` from our [google drive](https://drive.google.com/drive/folders/1UXxhAXK1MTMsKBNRyCM0CLKD_mLsskm4?usp=sharing) and place them in the ```Custom GloVe``` directory. These files were created by running the provided code (```build_vocab.sh``` ```cut_vocab.sh``` ```pickle_vocab.py``` ```cooc.py``` on the cleaned dataset: ```clean_pos_train_with_stopWords.txt``` and ```clean_neg_train_with_stopWords.txt```.

Finally, you need to run the ```Custom_GloVe.ipynb``` notebook. This notebook will output the ```embeddings_index.pkl``` file; the pickle file containing the embedding.

## To reproduce our best score :
You need to run the run.py file. If you don't have the required files it will download them automatically from the aforementioned drive.


