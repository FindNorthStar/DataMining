import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
comb_df = train_df.append(test_df)
members_df = pd.read_csv('data/members.csv')
songs_df = pd.read_csv('data/songs.csv')
song_extra_info_df = pd.read_csv('data/song_extra_info.csv')

songs_df = songs_df.merge(song_extra_info_df, on='song_id', how='left')
songs_df['song_year'] = songs_df['isrc'].apply(transform_isrc_to_year)
songs_df['song_country'] = songs_df['isrc'].apply(transform_isrc_to_country)

train_df = train_df.merge(songs_df, on='song_id', how='left')
test_df = test_df.merge(songs_df, on='song_id', how='left')

train_df = train_df.merge(members_df, on='msno', how='left')
test_df = test_df.merge(members_df, on='msno', how='left')

# fill missing data to visualize
train_df.fillna('nan', inplace=True)
test_df.fillna('nan', inplace=True)


def parse_splitted_category_to_number(x):
    if x is np.nan:
        return 0

    x = str(x)
    x.replace('/', '|')
    x.replace(';', '|')
    x.replace('\\', '|')
    x.replace('&', '|')
    x.replace(' and ', '|')
    x.replace('+', '|')
    return x.count('|') + 1


train_df['genre_count'] = train_df['genre_ids'].apply(parse_splitted_category_to_number)
train_df['artist_count'] = train_df['artist_name'].apply(parse_splitted_category_to_number)
train_df['composer_count'] = train_df['composer'].apply(parse_splitted_category_to_number)
train_df['lyricist_count'] = train_df['lyricist'].apply(parse_splitted_category_to_number)

count_df = train_df[
    ['song_id', 'genre_ids', 'genre_count', 'artist_name', 'artist_count', 'composer', 'composer_count', 'lyricist',
     'lyricist_count', 'target']]

count_df.head(10)


song_length_unique_df = train_df[['song_id', 'song_length']].drop_duplicates('song_id')

song_length_unique_series = song_length_unique_df['song_length'].astype(np.float64)

# assert(song_length_unique_series.duplicates())
# print(song_length_unique_df)
sns.boxplot(x=song_length_unique_series)
plt.show()

song_length_unique_series.describe().astype(np.int64)

song_length_unique_series = song_length_unique_series[np.abs(song_length_unique_series-song_length_unique_series.mean()) <= (3 * song_length_unique_series.std())]
song_length_unique_series.describe().astype(np.int64)


sns.boxplot(song_length_unique_series)
plt.show()

sns.distplot(song_length_unique_series)
plt.show()

count_df = train_df[['song_length', 'target']]
count_df = count_df[count_df['target'] == 1]


# transform scale unit into "log10"
count_df['song_length'] = count_df['song_length'].astype(np.float64)
print(count_df['song_length'].min(), count_df['song_length'].max(), count_df['song_length'].mean())
length_bins = np.logspace(np.log10(count_df['song_length'].min()), np.log10(count_df['song_length'].max()), 100)

# ignore missing data
count_df['song_length'].fillna(0.0, inplace=True)

count_df['cut_song_length'] = pd.cut(count_df['song_length'], bins=length_bins)
count_df.head(10)

plt.figure(figsize=(30, 15))
plt.xticks(rotation=90)
g = sns.countplot(x='cut_song_length', data=count_df)
g.set_yscale('log', nonposy='clip')
plt.show()


count_df = train_df[['song_length', 'target']].groupby('song_length').agg('mean')
count_df.reset_index(inplace=True)
count_df.columns = ['song_length', 'replay_pb']

# transform scale unit into "log10"
count_df['song_length'] = count_df['song_length'].astype(np.float64)
length_bins = np.logspace(np.log10(count_df['song_length'].min()), np.log10(count_df['song_length'].max()), 100)
count_df['song_length'].fillna(0.0, inplace=True)

count_df['cut_song_length'] = pd.cut(count_df['song_length'], bins=length_bins)


count_df = count_df[['cut_song_length', 'replay_pb']].groupby('cut_song_length').agg('mean')
count_df.reset_index(inplace=True)
count_df.head(10)

plt.figure(figsize=(30, 15))
plt.xticks(rotation=90)
sns.barplot(y='replay_pb', x='cut_song_length', data=count_df)
plt.show()


count_df = train_df[['genre_count', 'target']].groupby('genre_count').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['genre_count', 'replay_pb', 'play_count']

count_df['replay_count'] = (count_df['replay_pb'] * count_df['play_count']).astype(np.int32)

sns.barplot(x='genre_count', y='replay_pb', data=count_df)
plt.show()

g = sns.barplot(x='genre_count', y='replay_count', data=count_df)
g.set_yscale('log', nonposy='clip')
plt.show()
count_df

count_df = train_df[['genre_ids', 'target']].groupby('genre_ids').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['genre_ids', 'replay_pb', 'play_count']
print(len(count_df))

count_df['replay_count'] = (count_df['replay_pb'] * count_df['play_count']).astype(np.int32)
count_df = count_df.sort_values(by=['replay_count'], ascending=False)

plt.figure(figsize=(15, 80))
g = sns.barplot(y='genre_ids', x='replay_count', data=count_df)
g.set_xscale('log', nonposx='clip')
plt.show()

artist_df = train_df[['artist_name', 'target']].groupby('artist_name').agg(['mean', 'count']).reset_index()
artist_df.columns = ['artist_name', 'replay_pb', 'play_count']
# artist_df['artist_name'] = artist_df['artist_name'].apply(lambda x: x.decode('utf-8'))
artist_df['replay_count'] = (artist_df['replay_pb'] * artist_df['play_count']).astype(np.int32)
artist_df.head(15)

sns.barplot(y='artist_name', x='play_count', data=artist_df.sort_values(by=['play_count'], ascending=False).head(20))
plt.show()

sns.barplot(y='artist_name', x='replay_count', data=artist_df.sort_values(by=['replay_count'], ascending=False).head(20))
plt.show()

sns.barplot(y='artist_name', x='replay_pb', data=artist_df.sort_values(by=['replay_count'], ascending=False).head(20))
plt.show()

count_df = train_df[['composer_count', 'target']].groupby('composer_count').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['composer_count', 'replay_pb', 'play_count']

count_df['replay_count'] = (count_df['replay_pb'] * count_df['play_count']).astype(np.int32)

sns.barplot(x='composer_count', y='replay_pb', data=count_df)
plt.show()

g = sns.barplot(x='composer_count', y='replay_count', data=count_df)
g.set_yscale('log', nonposy='clip')
plt.show()
count_df

composer_df = train_df[['composer', 'target']].groupby('composer').agg(['mean', 'count']).reset_index()
composer_df.columns = ['composer', 'replay_pb', 'play_count']
composer_df['replay_count'] = (composer_df['replay_pb'] * composer_df['play_count']).astype(np.int32)


sns.barplot(y='composer', x='play_count', data=composer_df.sort_values(by=['play_count'], ascending=False).head(20))
plt.show()

sns.barplot(y='composer', x='replay_count', data=composer_df.sort_values(by=['replay_count'], ascending=False).head(20))
plt.show()

sns.barplot(y='composer', x='replay_pb', data=composer_df.sort_values(by=['replay_count'], ascending=False).head(20))
plt.show()

print(composer_df['play_count'].corr(composer_df['replay_pb']))
print(composer_df['play_count'].corr(composer_df['replay_count']))

composer_df.sort_values(by=['play_count'], ascending=False).head(15)

count_df = train_df[['lyricist_count', 'target']].groupby('lyricist_count').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['lyricist_count', 'replay_pb', 'play_count']

count_df['replay_count'] = (count_df['replay_pb'] * count_df['play_count']).astype(np.int32)

sns.barplot(x='lyricist_count', y='replay_pb', data=count_df)
plt.show()

g = sns.barplot(x='lyricist_count', y='replay_count', data=count_df)
g.set_yscale('log', nonposy='clip')
plt.show()
count_df

lyricist_df = train_df[['lyricist', 'target']].groupby('lyricist').agg(['mean', 'count']).reset_index()
lyricist_df.columns = ['lyricist', 'replay_pb', 'play_count']
lyricist_df['replay_count'] = (lyricist_df['replay_pb'] * lyricist_df['play_count']).astype(np.int32)


g = sns.barplot(y='lyricist', x='play_count', data=lyricist_df.sort_values(by=['play_count'], ascending=False).head(20))
g.set_xscale('log', nonposx='clip')
plt.show()

g = sns.barplot(y='lyricist', x='replay_count', data=lyricist_df.sort_values(by=['replay_count'], ascending=False).head(20))
g.set_xscale('log', nonposx='clip')
plt.show()

sns.barplot(y='lyricist', x='replay_pb', data=lyricist_df.sort_values(by=['replay_count'], ascending=False).head(20))
plt.show()

print(lyricist_df['play_count'].corr(lyricist_df['replay_pb']))
print(lyricist_df['play_count'].corr(lyricist_df['replay_count']))

lyricist_df.sort_values(by=['play_count'], ascending=False).head(15)

language_df = train_df[['language', 'target']].groupby('language').agg(['mean', 'count']).reset_index()
language_df.columns = ['language', 'replay_pb', 'play_count']
language_df['replay_count'] = (language_df['replay_pb'] * language_df['play_count']).astype(np.int32)

print(language_df.info())
g = sns.barplot(x='language', y='play_count', data=language_df.sort_values(by=['play_count'], ascending=False))
g.set_yscale('log', nonposy='clip')
plt.show()

g = sns.barplot(x='language', y='replay_count', data=language_df.sort_values(by=['replay_count'], ascending=False))
g.set_yscale('log', nonposy='clip')
plt.show()

sns.barplot(x='language', y='replay_pb', data=language_df.sort_values(by=['replay_count'], ascending=False))
plt.show()

print(language_df['replay_count'].corr(language_df['replay_pb']))
print(language_df['play_count'].corr(language_df['replay_pb']))
print(language_df['play_count'].corr(language_df['replay_count']))

g = sns.jointplot(x="play_count", y="replay_pb", data=language_df, kind="reg")
plt.show()
language_df.sort_values(by=['play_count'], ascending=False)


songs_df.isnull().sum() / songs_df.shape[0] * 100

count_df = songs_df[['song_id', 'song_year']].groupby('song_year').agg('count').reset_index()
count_df.columns = ['song_year', 'count']

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
sns.pointplot(x='song_year', y='count', data=count_df)
plt.show()

song_year_df = train_df[['song_year', 'target']].groupby('song_year').agg(['mean', 'count']).reset_index()
song_year_df.columns = ['song_year', 'replay_pb', 'play_count']
song_year_df['replay_count'] = (song_year_df['replay_pb'] * song_year_df['play_count']).astype(np.int32)

print(song_year_df.info())
plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.barplot(x='song_year', y='play_count', data=song_year_df.sort_values(by=['play_count'], ascending=False))
g.set_yscale('log', nonposy='clip')
plt.show()

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.barplot(x='song_year', y='replay_count', data=song_year_df.sort_values(by=['replay_count'], ascending=False))
g.set_yscale('log', nonposy='clip')
plt.show()

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
sns.barplot(x='song_year', y='replay_pb', data=song_year_df.sort_values(by=['replay_count'], ascending=False))
plt.show()

print(song_year_df['replay_count'].corr(song_year_df['replay_pb']))
print(song_year_df['play_count'].corr(song_year_df['replay_pb']))
print(song_year_df['play_count'].corr(song_year_df['replay_count']))

sns.jointplot(x="play_count", y="replay_pb", data=song_year_df, kind="reg", ylim=(0 ,1))
plt.show()
song_year_df.sort_values(by=['play_count'], ascending=False)

song_country_df = train_df[['song_country', 'target']].groupby('song_country').agg(['mean', 'count']).reset_index()
song_country_df.columns = ['song_country', 'replay_pb', 'play_count']
song_country_df['replay_count'] = (song_country_df['replay_pb'] * song_country_df['play_count']).astype(np.int32)

print(song_country_df.info())
plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.barplot(x='song_country', y='play_count', data=song_country_df.sort_values(by=['play_count'], ascending=False))
g.set_yscale('log', nonposy='clip')
plt.show()

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.barplot(x='song_country', y='replay_count', data=song_country_df.sort_values(by=['replay_count'], ascending=False))
g.set_yscale('log', nonposy='clip')
plt.show()

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
sns.barplot(x='song_country', y='replay_pb', data=song_country_df.sort_values(by=['replay_count'], ascending=False))
plt.show()

print(song_country_df['replay_count'].corr(song_country_df['replay_pb']))
print(song_country_df['play_count'].corr(song_country_df['replay_pb']))
print(song_country_df['play_count'].corr(song_country_df['replay_count']))

g = sns.jointplot(x="play_count", y="replay_pb", data=song_country_df, kind="reg")
plt.show()
song_country_df.sort_values(by=['play_count'], ascending=False)

count_df = train_df[['song_id', 'target']].groupby('song_id').agg(['mean', 'count']).reset_index()
count_df.columns = ['song_id', 'replay_pb', 'play_count']
count_df = count_df[count_df['play_count'] > 3000]

g = sns.jointplot(x="replay_pb", y="play_count", data=count_df, kind="reg")
plt.show()

