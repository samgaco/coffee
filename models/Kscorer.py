import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer



class Rscorer():

	def __init__(self):

		self.model = load_model("/home/samuel/Work/coffe2/8epoch.sav")


	def score(self, reviews):

		df = pd.DataFrame.from_dict(reviews, orient='index')

		df_tok = self.preprocess_R(df)

		return self.predict_R(df_tok)


	def predict_R(self, df_tok, df):

		pred = self.model.predict(df_tok)

		pred = (pred * [1, 2, 3, 4, 5]).sum(axis=1)

		df["pred"] = pred

		df_pred = df.groupby('business_id')['pred'].agg(
			[pd.np.min, pd.np.max, pd.np.mean])

		return pred, df_pred

	def preprocess_R(self, df):

		df = df["text"]

		max_features = 20000
		tokenizer = Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(list(df))

		list_tokenized_df = tokenizer.texts_to_sequences(df)

		# adjust length of sentences
		maxlen = 250
		df_tok = pad_sequences(list_tokenized_df, maxlen=maxlen)

		return df_tok
