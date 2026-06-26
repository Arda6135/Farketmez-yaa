import pandas as pd
import numpy as np

class DataSetPreparer:

    def __init__(self):
        self.imdb_path = './imdb.csv'
        self.tweets_path = './tweets.csv'

    def prepareDataSet(self, mode = 'both'):

        if mode == 'imdb':
            df_imdb = pd.read_csv(self.imdb_path)
            df_imdb.columns = ['text','label']
            df_imdb = df_imdb.dropna()

            imdb_mapping = {
                'negative': 0, 'Negative': 0, 0: 0,
                'positive': 1, 'Positive': 1, 1: 1,
            }

            x_raw = df_imdb['text']
            y= df_imdb['label'].map(imdb_mapping)

            return x_raw, y


        elif mode == 'tweets':

            df_tweet = pd.read_csv(self.tweets_path)
            df_tweet_reduced = df_tweet.drop(columns=['textID','selected_text'])
            df_tweet_reduced.columns = ['text','label']
            df_tweet_reduced = df_tweet_reduced.dropna()

            tweet_mapping = {
                'negative': 0, 'Negative': 0, 0: 0,
                'neutral': 1, 'Neutral': 1, 2: 1,
                'positive': 2, 'Positive': 2, 1: 2
            }

            x_raw = df_tweet_reduced['text']
            y = df_tweet_reduced['label'].map(tweet_mapping)

            return x_raw, y
        
        else:
            df_imdb = pd.read_csv(self.imdb_path)
            df_tweet = pd.read_csv(self.tweets_path)
            df_tweet_reduced = df_tweet.drop(columns=['textID','selected_text'])
            df_imdb.columns = ['text','label']
            df_tweet_reduced.columns = ['text','label']

            df_concat = pd.concat([df_imdb, df_tweet_reduced]).dropna()

            fallback_mapping = {
                'negative': 0,
                'neutral': 1,
                'positive': 2
                }
            
            x_raw = df_concat['text']
            y = df_concat['label'].map(fallback_mapping)

            return x_raw, y

  
        