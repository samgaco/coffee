import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class LRRscorer():

    def __init__(self):


        self.model_one = pickle.load(open( "/home/samuel/Work/coffe2/02_model/models_saved/LR/models/big_set/ lr_one1percent_test.sav", 'rb'))
        self.model_two = pickle.load(open("/home/samuel/Work/coffe2/02_model/models_saved/LR/models/big_set/ lr_two1percent_test.sav", 'rb'))
        self.model_three = pickle.load(open("/home/samuel/Work/coffe2/02_model/models_saved/LR/models/big_set/ lr_three1percent_test.sav", 'rb'))
        self.model_four = pickle.load(open("/home/samuel/Work/coffe2/02_model/models_saved/LR/models/big_set/ lr_four1percent_test.sav", 'rb'))
        self.model_five = pickle.load(open("/home/samuel/Work/coffe2/02_model/models_saved/LR/models/big_set/ lr_five1percent_test.sav", 'rb'))

        self.model_array = [self.model_one, self.model_two, self.model_three, self.model_four, self.model_five ]


    def score(self, reviews, number):


        df_tok = self.preprocess_R(reviews)

        return self.predict_R(df_tok, reviews, number)


    def predict_R(self, df_tok, df, number):

        pred = self.model_array[number].predict_proba(df_tok)[:, 1]

        df["pred"] = pred

        df_pred1 = df.groupby('business_id')['pred'].agg([pd.np.min, pd.np.max, pd.np.median, pd.np.std, pd.np.mean])

        return pred, df_pred1

    def preprocess_R(self, df):

        vectorizer = pickle.load(open("/home/samuel/Work/coffe2/02_model/models_saved/LR/data/vectorizer.sav", 'rb'))

        df["text"] = df["text"].astype(str)
        df_tok = vectorizer.transform(df["text"])


        return df_tok
