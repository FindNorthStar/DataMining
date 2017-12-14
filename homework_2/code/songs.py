import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime as dt

import json
import os.path
import gc

MODEL_FILE_NAME = 'model.txt'

# Load data
songs_df = pd.read_csv('data/songs.csv')
songs_df.info()
songs_df.head(15)
songs_df.isnull().sum() / songs_df.shape[0] * 100

print(len(songs_df['artist_name'].unique()))

def parse_splitted_category_to_number(x):
    return 0if x is np.nan else len(str(x).split('|'))


songs_df['genre_count'] = songs_df['genre_ids'].apply(parse_splitted_category_to_number)
songs_df['composer_count'] = songs_df['composer'].apply(parse_splitted_category_to_number)
songs_df['lyricist_count'] = songs_df['lyricist'].apply(parse_splitted_category_to_number)
songs_df.head(15)

train_df = pd.read_csv('data/train.csv')
train_df = train_df[['msno', 'song_id', 'target']]
train_df

count_df = train_df[['song_id', 'target']].groupby('song_id').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['song_id', 'play_prob', 'play_count']
count_df['play_recur'] = (count_df['play_prob'] * count_df['play_count']).astype(np.int16)

train_df = train_df.merge(count_df, on='song_id', how='left')
train_df

