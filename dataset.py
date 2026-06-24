import pandas as pd
import numpy as np

class DataSetPreparer:

    def __init__(self):
        self.imdb_path = './imbd.csv'
        self.tweets_path = './tweets.csv'

    def prepareDataSet(self):


        df_imdb = pd.read_csv(self.imdb_path)
        df_tweet = pd.read_csv(self.tweets_path)

        df_tweet_reduced = df_tweet.drop(columns=['textID','selected_text'])

        df_imdb.columns = ['text','label']
        df_tweet_reduced.columns = ['text','label']

        df_concat = pd.concat([df_imdb,df_tweet_reduced])

        x_raw = df_concat['text']
        y = df_concat['label']

        return x_raw,y