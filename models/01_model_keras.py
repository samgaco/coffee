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
from keras.callbacks import TensorBoard
import time
from keras.models import load_model


#read data
cwd = os.getcwd()

bigdataset = True

if bigdataset:

    fname = cwd + "/data/coffee_data_reviews_heavy.csv"
    fname2 = "/home/samuel/Work/coffe2/data/coffee_data_heavyscores.csv"

else:

    fname = "/home/samuel/Work/coffe/data/coffee_data_reviews.csv"
    fname2 = "/home/samuel/Work/coffe/data/coffee_data.csv"



df = pd.read_csv(fname)
df2 = pd.read_csv(fname2)

score = df2["score"]
mergeid = pd.DataFrame(score)
mergeid["business_id"] = df2["business_id"]


data_mer = pd.merge( mergeid, df, on="business_id")



#creamos variable respuesta
y = data_mer['score']
y =  pd.get_dummies(y.values)
y.columns = ["one", "two", "three", "four", "five"]


data_mer["text"] = data_mer["text"].astype(str)

""" 

#dividimos las explicativas y respuestas con un ratio 30/70, entre test set
#y set de entrenamiento

"""


x_validation = data_mer.iloc[0:60000,]
y_validation = y.iloc[0:60000,]

data_mer = data_mer.iloc[60000:data_mer.shape[0],]
y = y.iloc[60000:y.shape[0],]

random.seed(123)

random.seed(123)
x_train, x_test, y_train, y_test = train_test_split(data_mer[['text', "business_id"]],
                                                    y, test_size=0.05,
                                                    random_state=53)



x_train_id = x_train
x_test_id = x_test
x_validation_id = x_validation

x_train = x_train["text"]
x_test = x_test["text"]
x_validation = x_validation["text"]



"""

## We will index the words now and use a LSTM to train them
#Tokenization, Indexing, Index Representation


"""


max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
list_tokenized_train = tokenizer.texts_to_sequences(x_train)
list_tokenized_test = tokenizer.texts_to_sequences(x_test)
list_tokenized_val = tokenizer.texts_to_sequences(x_validation)


#adjust length of sentences
maxlen = 300
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
X_val = pad_sequences(list_tokenized_val , maxlen=maxlen)


#we get a plot with the review count and number of words in each
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.show()


"""

#model creation starts here

"""

inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier

embed_size = 128

x = Embedding(max_features, embed_size)(inp)

x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(5, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

batch_size = 32
epochs = 11

"# This line is for visualizing the model fit in tensorboard"


tensorboard = TensorBoard(log_dir="./logs", histogram_freq=1, batch_size=32)

model.fit(X_t,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tensorboard])

model.save("/home/samuel/Work/coffe2/11epoch.h5")

"""

#Prediction of the values we input

"""


predicciones = model.predict(X_te)
predicciones_val = model.predict(X_val)

# VAL


VAL = True


"""

#We look for the KPI

"""



if VAL:
    predicciones = predicciones_val


    predicted_y_test = predicciones


    predicciones = (predicciones*[1,2,3,4,5]).sum(axis=1)
    y_test_c = (y_validation.values*[1,2,3,4,5]).sum(axis=1)


    mean_squared_error(y_test_c, predicciones)

    x_validation_id["predicciones"] = predicciones
    x_validation_id_agg = x_validation_id.groupby('business_id')['predicciones'].agg([pd.np.min, pd.np.max, pd.np.mean])


    xtest_merge = pd.merge(x_test_id, x_validation_id_agg, on="business_id")

    merge_final = pd.merge(xtest_merge,df2, on="business_id")


    mean_squared_error(merge_final["score"],merge_final["mean"])

    ## accuracies
    y_pred =  pd.get_dummies(predicciones.round())
    predicted_y_test_bkp = predicted_y_test.copy()
    predicted_y_test = y_pred


    print("First: \n{}".format(confusion_matrix(y_test['one'], predicted_y_test.iloc[:,0])))
    print("\nsecond: \n{}".format(confusion_matrix(y_test['two'], predicted_y_test.iloc[:,1])))
    print("\nthird: \n{}".format(confusion_matrix(y_test['three'], predicted_y_test.iloc[:,2])))
    print("\nfourth: \n{}".format(confusion_matrix(y_test['four'], predicted_y_test.iloc[:,3])))
    print("\nfifth: \n{}".format(confusion_matrix(y_test['five'], predicted_y_test.iloc[:,4])))

    print("\nFirst: \n{}".format(classification_report(y_test['one'], predicted_y_test.iloc[:,0])))
    print("\nSecond: \n{}".format(classification_report(y_test['two'], predicted_y_test.iloc[:,1])))
    print("\nThird: \n{}".format(classification_report(y_test['three'], predicted_y_test.iloc[:,2])))
    print("\nFourth: \n{}".format(classification_report(y_test['four'], predicted_y_test.iloc[:,3])))
    print("\nFifth: \n{}".format(classification_report(y_test['five'], predicted_y_test.iloc[:,4])))




else:

    x_validation_id["predicciones"] = predicciones
    x_test_id_agg = x_validation_id.groupby('business_id')['predicciones'].agg([pd.np.min, pd.np.max, pd.np.mean])


    xtest_merge = pd.merge(x_validation_id, x_test_id_agg, on="business_id")

    merge_final = pd.merge(xtest_merge,df2, on="business_id")


    mean_squared_error(merge_final["score_y"],merge_final["mean"])


    #saving the model
    filename = 'finalized_model.sav'
    joblib.dump(model, "/home/samuel/Work/coffe2/data/" + filename)

    print("First: \n{}".format(confusion_matrix(y_test['one'], predicted_y_test.iloc[:,0])))
    print("\nsecond: \n{}".format(confusion_matrix(y_test['two'], predicted_y_test.iloc[:,1])))
    print("\nthird: \n{}".format(confusion_matrix(y_test['three'], predicted_y_test.iloc[:,2])))
    print("\nfourth: \n{}".format(confusion_matrix(y_test['four'], predicted_y_test.iloc[:,3])))
    print("\nfifth: \n{}".format(confusion_matrix(y_test['five'], predicted_y_test.iloc[:,4])))

    print("\nFirst: \n{}".format(classification_report(y_test['one'], predicted_y_test.iloc[:,0])))
    print("\nSecond: \n{}".format(classification_report(y_test['two'], predicted_y_test.iloc[:,1])))
    print("\nThird: \n{}".format(classification_report(y_test['three'], predicted_y_test.iloc[:,2])))
    print("\nFourth: \n{}".format(classification_report(y_test['four'], predicted_y_test.iloc[:,3])))
    print("\nFifth: \n{}".format(classification_report(y_test['five'], predicted_y_test.iloc[:,4])))

