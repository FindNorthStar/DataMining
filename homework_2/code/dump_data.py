import seaborn as sns
import pandas as pd
import numpy as np

def print_missing_ratio_df(df):
    print('>> missing ratio:')
    print(df.isnull().sum() / df.shape[0] * 100)


train_df = pd.read_csv('data/train.csv')
train_df.info()
print_missing_ratio_df(train_df)
print(train_df.head(15))

test_df = pd.read_csv('data/test.csv')
test_df.info()
print_missing_ratio_df(test_df)
test_df.head(15)

songs_df = pd.read_csv('data/songs.csv')
songs_df.info()
print_missing_ratio_df(songs_df)
songs_df.head(15)

song_extra_info_df = pd.read_csv('data/song_extra_info.csv')
song_extra_info_df.info()
print_missing_ratio_df(song_extra_info_df)
song_extra_info_df.head(15)

members_df = pd.read_csv('data/members.csv')
members_df.info()
print_missing_ratio_df(members_df)
members_df.head(15)

