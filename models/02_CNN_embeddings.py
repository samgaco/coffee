import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import random
import nltk
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
import string
from gensim.models import Word2Vec
import numpy
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.merge import Add
from keras.layers.merge import Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
import gensim
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import codecs
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Reshape, Flatten
from keras.models import load_model
from keras.layers import embeddings


def RNN(rnn_size,  vocab_size, learning_rate):

    model = Sequential()
    model.add(LSTM(rnn_size, input_shape=( vocab_size,)))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    # adam optimizer
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model





# read data
cwd = os.getcwd()
fname = "/home/samuel/Work/coffe/data/coffee_data_reviews.csv"
fname2 = "/home/samuel/Work/coffe/data/coffee_data.csv"

df = pd.read_csv(fname)
df2 = pd.read_csv(fname2)

score = df2["score"]
mergeid = pd.DataFrame(score)
mergeid["business_id"] = df2["business_id"]

data_mer = pd.merge(mergeid, df, on="business_id")

# creamos variable respuesta
y = data_mer['score']
y = pd.get_dummies(y.values)
y.columns = ["one", "two", "three", "four", "five"]

data_mer["text"] = data_mer["text"].astype(str)

# dividimos las explicativas y respuestas con un ratio 30/70, entre test set
# y set de entrenamiento


x_validation = data_mer.iloc[0:60000, ]
y_validation = y.iloc[0:60000, ]

data_mer = data_mer.iloc[60000:data_mer.shape[0], ]
y = y.iloc[60000:y.shape[0], ]

random.seed(123)

random.seed(123)
x_train, x_test, y_train, y_test = train_test_split(data_mer[['text', "business_id"]],
                                                    y, test_size=0.30,
                                                    random_state=53)

x_train_id = x_train
x_test_id = x_test
x_validation_id = x_validation

x_train = x_train["text"]
x_test = x_test["text"]
x_validation = x_validation["text"]

"""

#We will create the word embeddings here

"""

nltk.download("stopwords")

useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)


def baglis(words):
    bag = []
    for i in range(len(words)):
        sentence = []
        for j in range(len(str(words.iloc[i]).split())):
            if str(words.iloc[i]).split()[j] not in useless_words:
                sentence.append(str(words.iloc[i]).split()[j])

        bag.append(sentence)

    return (bag)


bag = baglis(x_train)

word_embb = Word2Vec(bag)
word_embb.train(bag, total_examples=len(bag), epochs=2)

""""

#Create tensor of sentences (coming from the word embeddings)

"""

pos = 0
batch_size = 30

while  pos + batch_size < batch_size*2:

    batch = bag[pos:pos+batch_size]
    ybatch = y_train.iloc[pos:pos+batch_size,]

    xtrain_sentences = numpy.zeros((len(batch), 150, 100))

    for j in range(len(batch)):

        sentence = numpy.array(())

        for word in batch[j]:

            try:

                if len(sentence) == 0:
                    sentence = word_embb.wv[word]
                    sentence = np.reshape(sentence, (1, 100))
                else:
                    word_now = word_embb.wv[word]
                    sentence = np.vstack((sentence, word_now))

                if sentence.shape[0] == 150:
                    break

            except KeyError as e:
                print("Ignoring:", word)

        if (sentence.shape[0] == 0): continue

        sentence_pad = np.pad(sentence, ((0, 150 - sentence.shape[0]), (0, 0)), "constant")
        xtrain_sentences[j,] = sentence_pad




    xtrain_sentences = xtrain_sentences.reshape(batch_size,-1)



    pos += batch_size

# model creation starts here



    if pos == 0:
        model = CNN(xtrain_sentences.shape)
    else:
        model = load_model("/home/samuel/Work/coffe2/02_model/models_saved/CNN/CNN.hf5")

    callbacks = [EarlyStopping(monitor='val_loss')]
    model.fit(xtrain_sentences, ybatch,  batch_size=10, epochs=10, verbose=1, callbacks=callbacks)
    model.save("/home/samuel/Work/coffe2/02_model/models_saved/CNN/CNN.h5")
    del model