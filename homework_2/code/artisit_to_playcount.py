import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_splitted_category(x):
    if x is np.nan:
        return 'nan'
    x = str(x)
    x=x.replace('/', '|')
    x=x.replace(';', '|')
    x=x.replace('\\', '|')
    x=x.replace('&', '|')
    x=x.replace(' and ', '|')
    x=x.replace('AKA', '|')
    x=x.replace('.', '|')
    x = x.replace('、', '|')
    x = x.replace('／', '|')
    x=x.replace('+', '|')
    return x

train_df = pd.read_csv('data/train.csv')
songs_df = pd.read_csv('data/songs.csv')

songs_df['artist_name'] = songs_df['artist_name'].apply(parse_splitted_category)
songs_df['composer'] = songs_df['composer'].apply(parse_splitted_category)
songs_df['lyricist'] = songs_df['lyricist'].apply(parse_splitted_category)


train_df = train_df.merge(songs_df, on='song_id', how='left')
train_df.fillna('nan', inplace=True)


artist_df = train_df[['artist_name', 'target']].groupby('artist_name').agg(['mean', 'count']).reset_index()
artist_df.columns = ['artist_name', 'replay_pb', 'play_count']
artist_df['replay_count'] = (artist_df['replay_pb'] * artist_df['play_count']).astype(np.int32)
artist_df = artist_df.sort_values(by=["play_count"],ascending=True)
artist_df.reset_index().to_csv('data/artistId.csv')

composer_df = train_df[['composer', 'target']].groupby('composer').agg(['mean', 'count']).reset_index()
composer_df.columns = ['composer', 'replay_pb', 'play_count']
composer_df['replay_count'] = (composer_df['replay_pb'] * composer_df['play_count']).astype(np.int32)
composer_df = composer_df.sort_values(by=["play_count"],ascending=True)
composer_df.reset_index().to_csv('data/composerId.csv')

lyricist_df = train_df[['lyricist', 'target']].groupby('lyricist').agg(['mean', 'count']).reset_index()
lyricist_df.columns = ['lyricist', 'replay_pb', 'play_count']
lyricist_df['replay_count'] = (lyricist_df['replay_pb'] * lyricist_df['play_count']).astype(np.int32)
lyricist_df = lyricist_df.sort_values(by=["play_count"],ascending=True)
lyricist_df.reset_index().to_csv('data/lyricistId.csv')