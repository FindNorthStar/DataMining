import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def transform_isrc_to_year(isrc):
    if type(isrc) != str:
        return np.nan
    # this year 2017
    suffix = int(isrc[5:7])

    return 1900 + suffix if suffix > 17 else 2000 + suffix


def transform_isrc_to_country(isrc):
    if type(isrc) != str:
        return np.nan
    country = isrc[:2]

    return country


def parse_splitted_category_to_number(x):
    return x.count('|') + 1

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

def transform_nan_to_zero(x):
    if x == 'nan':
        return 0
    return x

def transform_nanyear_to_mean(x):
    if x == 'nan':
        return 2006.79360959945
    return x

def transform_nanId(x):
    if x == 'nan':
        return -1
    return x

def transform_gener(x):
    if x=='nan':
        return 0
    if "|" in x:
        return x[:x.find("|")].strip()
    return x

train_df = pd.read_csv('data/train.csv')
# test_df = pd.read_csv('data/test.csv')
# comb_df = train_df.append(test_df)

# members_df = pd.read_csv('data/members.csv')
songs_df = pd.read_csv('data/songs.csv')
song_extra_info_df = pd.read_csv('data/song_extra_info.csv')

songs_df = songs_df.merge(song_extra_info_df, on='song_id', how='left')
songs_df['song_year'] = songs_df['isrc'].apply(transform_isrc_to_year)
songs_df['genre_ids'] = songs_df['genre_ids'].apply(parse_splitted_category)
songs_df['genre_count'] = songs_df['genre_ids'].apply(parse_splitted_category_to_number)
songs_df['artist_name'] = songs_df['artist_name'].apply(parse_splitted_category)
songs_df['artist_count'] = songs_df['artist_name'].apply(parse_splitted_category_to_number)
songs_df['composer'] = songs_df['composer'].apply(parse_splitted_category)
songs_df['composer_count'] = songs_df['composer'].apply(parse_splitted_category_to_number)
songs_df['lyricist'] = songs_df['lyricist'].apply(parse_splitted_category)
songs_df['lyricist_count'] = songs_df['lyricist'].apply(parse_splitted_category_to_number)

count_df = train_df[['song_id', 'target']].groupby('song_id').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['song_id', 'replay_pb', 'play_count']
count_df['replay_count'] = (count_df['replay_pb'] * count_df['play_count']).astype(np.int32)

songs_df = songs_df.merge(count_df, on='song_id', how='left')
songs_df = songs_df.drop_duplicates('song_id')

songs_artist_df = pd.read_csv('data/artistId.csv')[['artist_name','artistId']]
songs_df = songs_df.merge(songs_artist_df, on='artist_name', how='left')
songs_composer_df = pd.read_csv('data/composerId.csv')[['composer','composerId']]
songs_df = songs_df.merge(songs_composer_df, on='composer', how='left')
songs_lyricist_df = pd.read_csv('data/lyricistId.csv')[['lyricist','lyricistId']]
songs_df = songs_df.merge(songs_lyricist_df, on='lyricist', how='left')

songs_df = songs_df[['song_id', 'genre_ids', 'genre_count', 'artistId', 'artist_count', 'composerId', 'composer_count', 'lyricistId', 'lyricist_count', 'song_year','language','song_length', 'play_count','replay_count', 'replay_pb']].drop_duplicates('song_id')


year_null = pd.isnull(songs_df['song_year'])
correct_mean_year = sum(songs_df['song_year'][year_null == False]) / len(songs_df['song_year'][year_null == False])
print(correct_mean_year)

songs_df.fillna('nan', inplace=True)
songs_df['play_count'] = songs_df['play_count'].apply(transform_nan_to_zero)
songs_df['replay_count'] = songs_df['replay_count'].apply(transform_nan_to_zero)
songs_df['replay_pb'] = songs_df['replay_pb'].apply(transform_nan_to_zero)
songs_df['song_year'] = songs_df['song_year'].apply(transform_nanyear_to_mean)
songs_df['genre_ids'] = songs_df['genre_ids'].apply(transform_gener)
songs_df['artistId'] = songs_df['artistId'].apply(transform_nanId)
songs_df['composerId'] = songs_df['composerId'].apply(transform_nanId)
songs_df['lyricistId'] = songs_df['lyricistId'].apply(transform_nanId)

songs_df.to_csv('data/temp_song.csv')
