import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
comb_df = train_df.append(test_df)
members_df = pd.read_csv('data/members.csv')
songs_df = pd.read_csv('data/songs.csv')
song_extra_info_df = pd.read_csv('data/song_extra_info.csv')

songs_df = songs_df.merge(song_extra_info_df, on='song_id', how='left')

train_df = train_df.merge(songs_df, on='song_id', how='left')
test_df = test_df.merge(songs_df, on='song_id', how='left')

train_df = train_df.merge(members_df, on='msno', how='left')
test_df = test_df.merge(members_df, on='msno', how='left')

# fill missing data to visualize
train_df.fillna('nan', inplace=True)
test_df.fillna('nan', inplace=True)


total_row_df = pd.DataFrame(data={
    'source': ['train', 'test', 'train + test'],
    'type': ['total', 'total', 'total'],
    'count': [train_df.shape[0],
              test_df.shape[0],
              comb_df.shape[0]]
})
sns.barplot(x='count', y='source', hue='type', data=total_row_df)
plt.show()

total_row_df

train_unique_msno_series = train_df['msno'].drop_duplicates()
test_unique_msno_series = test_df['msno'].drop_duplicates()
comb_unique_msno_series = comb_df['msno'].drop_duplicates()

msno_unique_df = pd.DataFrame(data={
    'source': ['train', 'test', 'train + test'],
    'type': ['unique', 'unique', 'unique'],
    'count': [len(train_unique_msno_series),
              len(test_unique_msno_series),
              len(comb_unique_msno_series)]
})
sns.barplot(x='count', y='source', hue='type', data=msno_unique_df)
plt.show()

print('%.2f%% unique users are contained in train set' % (len(train_unique_msno_series) / train_df.shape[0] * 100))
print('%.2f%% unique users are contained in test set' % (len(test_unique_msno_series) / test_df.shape[0] * 100))
print('%.2f%% unique users are contained in (train + test) set' % (len(comb_unique_msno_series) / comb_df.shape[0] * 100))
msno_unique_df

train_minus_test_by_msno_df = train_df[~train_df['msno'].isin(test_unique_msno_series)]
test_minus_train_by_msno_df = test_df[~test_df['msno'].isin(train_unique_msno_series)]

print('%d rows are contained in train set but not contained in test set' % len(train_minus_test_by_msno_df))
print('%d rows are contained in test set but not contained in train set' % len(test_minus_train_by_msno_df))

train_unique_song_series = train_df['song_id'].drop_duplicates()
test_unique_song_series = test_df['song_id'].drop_duplicates()
comb_unique_song_series = comb_df['song_id'].drop_duplicates()

song_unique_df = pd.DataFrame(data={
    'source': ['train', 'test', 'train + test'],
    'type': ['unique', 'unique', 'unique'],
    'count': [len(train_unique_song_series),
              len(test_unique_song_series),
              len(comb_unique_song_series)]
})
sns.barplot(x='count', y='source', hue='type', data=song_unique_df)
plt.show()

print('%.2f%% unique songs are contained in train set' % (len(train_unique_song_series) / train_df.shape[0] * 100))
print('%.2f%% unique songs are contained in test set' % (len(test_unique_song_series) / test_df.shape[0] * 100))
print('%.2f%% unique songs are contained in (train + test) set' % (len(comb_unique_song_series) / comb_df.shape[0] * 100))
song_unique_df

train_minus_test_by_song_df = train_df[~train_df['song_id'].isin(test_unique_song_series)]
test_minus_train_by_song_df = test_df[~test_df['song_id'].isin(train_unique_song_series)]

print('%d rows are contained in train set but not contained in test set' % len(train_minus_test_by_song_df))
print('%d rows are contained in test set but not contained in train set' % len(test_minus_train_by_song_df))

def target_count_plot(column):
    print(sorted(list(train_df[column].unique())))
    print(len(list(train_df[column].unique())))
    print(sorted(list(test_df[column].unique())))
    print(len(list(test_df[column].unique())))
    sns.countplot(y=column, hue='target', data=train_df, order=train_df[column].value_counts().index)
    plt.show()

target_count_plot('source_system_tab')

target_count_plot('source_screen_name')

target_count_plot('source_type')

train_df['source_merged'] = train_df['source_system_tab'].map(str) + ' | ' + train_df['source_screen_name'].map(str) + ' | ' + train_df['source_type'].map(str)
sns.countplot(y='source_merged', hue='target', data=train_df, order=train_df['source_merged'].value_counts().iloc[:20].index)
plt.show()

count_df = train_df[['source_merged', 'target']].groupby('source_merged').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['source_merged', 'source_replay_pb', 'source_replay_count']
count_df = count_df.sort_values(by='source_replay_pb', ascending=False)
count_df

