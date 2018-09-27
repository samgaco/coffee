from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import mglearn
from sklearn.ensemble import AdaBoostClassifier
import random
import pickle

# Read data
cwd = os.getcwd()

bigdataset = True

if bigdataset:

    fname = "/home/samuel/Work/coffe2/data/coffee_data_reviews_heavy.csv"
    fname2 = "/home/samuel/Work/coffe2/data/coffee_data_heavyscores.csv"

else:

    fname = "/home/samuel/Work/coffe/data/coffee_data_reviews.csv"
    fname2 = "/home/samuel/Work/coffe/data/coffee_data.csv"

df = pd.read_csv(fname)  # reviews data set
df2 = pd.read_csv(fname2)  # data set with original data

# Save score into a new variable
score = df2["score"]

# add Score to df (reviews) data set ==> mergeid is a dictionary with the scores and the business_id
mergeid = pd.DataFrame(score)
mergeid["business_id"] = df2["business_id"]
data_mer = pd.merge(mergeid, df, on="business_id")

# creamos variable respuesta
y = data_mer['score']
y = pd.get_dummies(y.values)
y.columns = ["one", "two", "three", "four", "five"]

# text column into a string
data_mer["text"] = data_mer["text"].astype(str)

# dividimos las explicativas y respuestas con un ratio 30/70, entre test set
# y set de entrenamiento

x_validation = data_mer.iloc[0:4000, ]
y_validation = y.iloc[0:4000, ]

data_mer = data_mer.iloc[4000:data_mer.shape[0], ]
y = y.iloc[4000:y.shape[0], ]

random.seed(123)
x_train, x_test, y_train, y_test = train_test_split(data_mer['text'],
                                                    y, test_size=0.01,
                                                    random_state=53)

##TfidVectorizer part DESCRIPTIVE
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), decode_error='ignore')
txt_fitted = vectorizer.fit(x_train)
X_train_word_features = txt_fitted.transform(x_train)
X = vectorizer.fit_transform(x_train)
test_features = vectorizer.transform(x_test)

idf = vectorizer.idf_
vectorizer.vocabulary_

rr = dict(zip(txt_fitted.get_feature_names(), idf))

token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()
token_weight.columns = ('token', 'weight')
token_weight = token_weight.sort_values(by='weight', ascending=False)
token_weight

tf = vectorizer
feature_names = np.array(tf.get_feature_names())
sorted_by_idf = np.argsort(tf.idf_)

print("Features with lowest idf: \n{}".format(
    feature_names[sorted_by_idf[:3]]
))

print("\nFeatures with highest idf:\n{}".format(
    feature_names[sorted_by_idf[-3:]]
))

new1 = tf.transform(x_train)
# find maximum value for each of the features over all of dataset:
max_val = new1.max(axis=0).toarray().ravel()
# sort weights from smallest to biggest and extract their indices
sort_by_tfidf = max_val.argsort()

print("Features with lowest tfidf:\n{}".format(
    feature_names[sort_by_tfidf[:3]]))

print("\nFeatures with highest tfidf: \n{}".format(
    feature_names[sort_by_tfidf[-3:]]))

##Applying model over 5 categories

# Models tried: Logistic regression, SVMC


class_names = ["one", "two", "three", "four", "five"]

losses = []
auc = []

y_predprobas = pd.DataFrame()

i = 0

for class_name in class_names:
    # call the labels one column at a time so we can run the classifier on them
    train_target = y_train[class_name]
    test_target = y_test[class_name]
    classifier = LogisticRegression(solver='sag', C=10)

    cv_loss = np.mean(cross_val_score(classifier, X_train_word_features, train_target, cv=5, scoring='neg_log_loss'))
    losses.append(cv_loss)
    print('CV Log_loss score for class {} is {}'.format(class_name, cv_loss))

    cv_score = np.mean(cross_val_score(classifier, X_train_word_features, train_target, cv=5, scoring='accuracy'))
    print('CV Accuracy score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(X_train_word_features, train_target)

    pkl_filename = " lr_" + class_name + "1percent_test" + ".sav"
    pickle.dump(classifier, open(pkl_filename, "wb"), protocol=2)

    y_pred = classifier.predict(test_features)
    y_pred_prob = classifier.predict_proba(test_features)[:, 1]

    y_predprobas[class_name] = y_pred_prob

    auc_score = metrics.roc_auc_score(test_target, y_pred_prob)
    auc.append(auc_score)
    print("CV ROC_AUC score {}\n".format(auc_score))

    print(confusion_matrix(test_target, y_pred))
    print(classification_report(test_target, y_pred))

print('Total average CV Log_loss score is {}'.format(np.mean(losses)))
print('Total average CV ROC_AUC score is {}'.format(np.mean(auc)))

y_predprobas.to_csv("/home/samuel/Work/coffe2/02_model/models_saved/LR/data/y_pred_proba.csv")

pkl_filename = "X_train_word_features"
pickle.dump(classifier, open(pkl_filename, "wb"), protocol=2)

pkl_filename = "vectorizer.sav"
pickle.dump(vectorizer, open("/home/samuel/Work/coffe2/02_model/models_saved/LR/data/" + pkl_filename, "wb"), protocol=2)