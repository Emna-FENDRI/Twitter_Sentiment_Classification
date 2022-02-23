from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.layers import Dropout, Masking, Embedding, Bidirectional, LSTM,Dense, Activation, GRU
from keras.models import Sequential
from helper import *
from embedding import *

    """
     5 Final Deep learning models
    """


def simple_nn(X_train, y_train, embedding_matrix,vocab_size,embedding_dim,maxlen, train = False, mask_zero=True):

    model = Sequential()
    model.add(layers.Embedding(vocab_size,embedding_dim,weights=[embedding_matrix],input_length=maxlen,trainable=True))    
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    if(train):
        history = model.fit(X_train, y_train,epochs=6,
                            verbose=True,
                            validation_split = 0.1,
                            batch_size=23)
        model.save("simple_nn")
        plot_history(history)
                  
    return keras.models.load_model("simple_nn")



def cnn (X_train, y_train,embedding_matrix,vocab_size,embedding_dim,maxlen, train = False):

    model = Sequential()
    model.add(layers.Embedding(vocab_size,embedding_dim,weights=[embedding_matrix],input_length=maxlen, trainable=True, mask_zero=True))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
                  
    if(train):
        history = model.fit(X_train, y_train,epochs=4,
                            verbose=True,
                            validation_split = 0.1,
                            batch_size=23)
        model.save("CNN")
        plot_history(history)
                  
    return keras.models.load_model("CNN")



def multi_cnn(X_train, y_train, embedding_matrix,vocab_size,embedding_dim,maxlen, train = False, mask_zero=True):

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.0)
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    inputs = keras.Input(shape=(100,))
    embedding_layer = layers.Embedding(vocab_size,200,weights=[embedding_matrix],input_length=100, trainable = True)(inputs)
    kernels = [3,4,5] 
    pools = []
    for k in kernels: 
        conv1D_layer = layers.Conv1D(500, k, activation='relu')(embedding_layer)
        maxPool = layers.GlobalMaxPooling1D()(conv1D_layer)
        pools.append(maxPool)

    poolFin = layers.concatenate(pools)

    mlp = Dropout(0.4)(poolFin)
    mlp = Dense(256, activation = 'relu')(mlp)
    mlp = Dropout(0.3)(mlp)
    mlp = Dense(128, activation = 'relu')(mlp)
    mlp = Dropout(0.4)(mlp)
    outputs = Dense(1, activation='sigmoid')(mlp)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    model.summary()


    if(train):
        history = model.fit(X_train, y_train,epochs=5,
                            verbose=True,
                            validation_split = 0.1,
                            batch_size=1024, callbacks=[reduce_lr, early_stop])
        model.save("MULTI_CNN")
        plot_history(history)
                  
    return keras.models.load_model("MULTI_CNN")    
    
    


def lstm(X_train, y_train ,embedding_matrix,vocab_size,embedding_dim,maxlen, train = False, mask_zero=True):
    
    model = Sequential()
    model.add(layers.Embedding(vocab_size, 
                                embedding_dim, 
                                weights=[embedding_matrix], 
                                input_length=maxlen, 
                                trainable=True))

    model.add(layers.LSTM(128,dropout=0.5))
    model.add(layers.Dense(64,activation='relu'))
    model.add(Dropout(0.4))
    model.add(layers.Dense(1,activation='sigmoid'))


    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    if(train):
        history = model.fit(X_train, y_train,epochs=8,
                            verbose=True,
                            validation_split = 0.1,
                            batch_size=23)
        model.save("LSTM")
        plot_history(history)
                  
    return keras.models.load_model("LSTM")

def bi_lstm(X_train, y_train ,embedding_matrix,vocab_size,embedding_dim,maxlen, train = False, mask_zero=True):

    model = Sequential()

    model.add(layers.Embedding(vocab_size, 
                                embedding_dim, 
                                weights=[embedding_matrix], 
                                input_length=maxlen, 
                                trainable=True))

    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
                  
    if(train):
        history = model.fit(X_train, y_train,epochs=6,
                            verbose=True,
                            validation_split = 0.1,
                            batch_size=23)
        model.save("BidirLSTM")
        plot_history(history)
                  
    return keras.models.load_model("BidirLSTM")

def ensemble_learning(y1,y2,y3):
    y = y1+y2+y3
    y[y==0]= 0
    y[y==1]= 0
    y[y==2]= 1
    y[y==3]= 1
    return y