import pandas as pd
import numpy as np
import seaborn as sns
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

city_df = train_df[['city', 'target']].groupby('city').agg(['mean', 'count']).reset_index()
city_df.columns = ['city', 'replay_pb', 'play_count']
city_df['replay_count'] = (city_df['replay_pb'] * city_df['play_count']).astype(np.int32)

g = sns.barplot(x='city', y='play_count', data=city_df, order=city_df.sort_values(by=['play_count'], ascending=False)['city'])
g.set_yscale('log', nonposy='clip')
plt.show()

g = sns.barplot(x='city', y='replay_count', data=city_df, order=city_df.sort_values(by=['replay_count'], ascending=False)['city'])
g.set_yscale('log', nonposy='clip')
plt.show()

sns.barplot(x='city', y='replay_pb', data=city_df, order=city_df.sort_values(by=['replay_count'], ascending=False)['city'])
plt.show()

print(city_df['replay_count'].corr(city_df['replay_pb']))
print(city_df['play_count'].corr(city_df['replay_pb']))
print(city_df['play_count'].corr(city_df['replay_count']))

# sns.jointplot(x="play_count", y="replay_pb", data=city_df, kind="reg")

city_df.sort_values(by=['play_count'], ascending=False)

bd_members_df = members_df[['msno', 'bd']]

sns.boxplot(y=bd_members_df['bd'])
plt.show()

bd_members_df = members_df[['msno', 'bd']]
# remove invalid values
bd_members_df = bd_members_df[(bd_members_df['bd'] > 7) & (bd_members_df['bd'] < 120)]
sns.boxplot(y=bd_members_df['bd'])
plt.show()

bd_mean = bd_members_df['bd'].mean()
bd_std = bd_members_df['bd'].std()

bd_df = train_df[['bd', 'target']].groupby('bd').agg(['mean', 'count']).reset_index()
bd_df.columns = ['bd', 'replay_pb', 'play_count']
bd_df['replay_count'] = (bd_df['replay_pb'] * bd_df['play_count']).astype(np.int32)

# remove outliers
bd_df = bd_df[np.abs(bd_df['bd'] - bd_mean) <= 3 * bd_std]
plt.figure(figsize=(30, 15))
g = sns.barplot(x='bd', y='play_count', data=bd_df)
g.set_yscale('log', nonposy='clip')
plt.show()

plt.figure(figsize=(30, 15))
g = sns.barplot(x='bd', y='replay_count', data=bd_df)
g.set_yscale('log', nonposy='clip')
plt.show()

bd_length_bins = [0, 12, 18, 22, 25, 30, 35, 45, 55, 65]
bd_df['cut_bd_length'] = pd.cut(bd_df['bd'], bins=bd_length_bins)
tmp_bd_df = bd_df.groupby('cut_bd_length').agg(['mean', 'sum']).reset_index()
tmp_bd_df.columns = ['bd_range', '_', '_', 'replay_pb', '_', '_', 'play_count', '_', 'replay_count']
tmp_bd_df.drop(['_'], axis=1, inplace=True)
tmp_bd_df

g = sns.barplot(x='bd_range', y='replay_count', data=tmp_bd_df)
g.set_yscale('log', nonposy='clip')
plt.show()

sns.barplot(x='bd_range', y='replay_pb', data=tmp_bd_df)
plt.show()

sns.countplot(x='gender', hue='target', data=train_df, order=train_df['gender'].value_counts().index)
plt.show()

count_df = train_df[['msno', 'gender']].drop_duplicates('msno')
sns.countplot(x='gender', data=count_df, order=count_df['gender'].value_counts().index)
plt.show()
print(count_df['gender'].value_counts())

count_df = members_df[['msno', 'gender']].fillna('nan').drop_duplicates('msno')
sns.countplot(x='gender', data=count_df, order=count_df['gender'].value_counts().index)
plt.show()
print(count_df['gender'].value_counts())

count_df = train_df[['msno', 'registered_via']].drop_duplicates('msno')
sns.countplot(x='registered_via', data=count_df, order=count_df['registered_via'].value_counts().index)
plt.show()
print(count_df['registered_via'].value_counts())

count_df = members_df[['msno', 'registered_via']].fillna('nan')
sns.countplot(x='registered_via', data=count_df, order=count_df['registered_via'].value_counts().index)
plt.show()
print(count_df['registered_via'].value_counts())

count_df = train_df[['registered_via', 'target']]
sns.countplot(x='registered_via', hue='target', data=count_df, order=count_df['registered_via'].value_counts().index)
plt.show()
print(count_df['registered_via'].value_counts())

def transform_init_time_to_ym(time):
    time_str = str(time)
    year = int(time_str[:4])
    month = int(time_str[4:6])
    return int("%04d%02d" % (year, month))

# Consider year, month.
count_df = train_df[['msno', 'registration_init_time']].drop_duplicates('msno')
count_df['registration_init_ym'] = count_df['registration_init_time'].apply(transform_init_time_to_ym)

tmp_count_df = count_df[['msno', 'registration_init_ym']].groupby('registration_init_ym').agg('count').reset_index()

tmp_count_df.columns = ['registration_init_ym', 'count']
tmp_count_df

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.pointplot(x='registration_init_ym', y='count', data=tmp_count_df)
plt.show()

# ===================================================================================================================
count_df = members_df[['msno', 'registration_init_time']]
count_df['registration_init_ym'] = count_df['registration_init_time'].apply(transform_init_time_to_ym)

tmp_count_df = count_df[['msno', 'registration_init_ym']].groupby('registration_init_ym').agg('count').reset_index()

tmp_count_df.columns = ['registration_init_ym', 'count']
tmp_count_df

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.pointplot(x='registration_init_ym', y='count', data=tmp_count_df)
plt.show()

count_df = train_df[['msno', 'registration_init_time', 'target']].copy()
count_df['registration_init_ym'] = count_df['registration_init_time'].apply(transform_init_time_to_ym)

reg_init_df = count_df[['registration_init_ym', 'target']].groupby('registration_init_ym').agg(['mean', 'count']).reset_index()
reg_init_df.columns = ['registration_init_ym', 'replay_pb', 'play_count']

reg_init_df['replay_count'] = (reg_init_df['replay_pb'] * reg_init_df['play_count']).astype(np.int32)

print(reg_init_df.info())
plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
sns.barplot(x='registration_init_ym', y='play_count', data=reg_init_df.sort_values(by=['play_count'], ascending=False))
plt.show()

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
sns.barplot(x='registration_init_ym', y='replay_count', data=reg_init_df.sort_values(by=['replay_count'], ascending=False))
plt.show()

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
sns.barplot(x='registration_init_ym', y='replay_pb', data=reg_init_df.sort_values(by=['replay_count'], ascending=False))
plt.show()

print(reg_init_df['replay_count'].corr(reg_init_df['replay_pb']))
print(reg_init_df['play_count'].corr(reg_init_df['replay_pb']))
print(reg_init_df['play_count'].corr(reg_init_df['replay_count']))

g = sns.jointplot(x="play_count", y="replay_pb", data=reg_init_df, kind="reg")
plt.show()
reg_init_df.sort_values(by=['play_count'], ascending=False)

# Consider year, month.
count_df = train_df[['msno', 'expiration_date']].drop_duplicates('msno')
count_df['expiration_date_ym'] = count_df['expiration_date'].apply(transform_init_time_to_ym)

tmp_count_df = count_df[['msno', 'expiration_date_ym']].groupby('expiration_date_ym').agg('count').reset_index()

tmp_count_df.columns = ['expiration_date_ym', 'count']
tmp_count_df

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.pointplot(x='expiration_date_ym', y='count', data=tmp_count_df)
plt.show()

# ===================================================================================================================
count_df = members_df[['msno', 'expiration_date']]
count_df['expiration_date_ym'] = count_df['expiration_date'].apply(transform_init_time_to_ym)

tmp_count_df = count_df[['msno', 'expiration_date_ym']].groupby('expiration_date_ym').agg('count').reset_index()

tmp_count_df.columns = ['expiration_date_ym', 'count']
tmp_count_df

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.pointplot(x='expiration_date_ym', y='count', data=tmp_count_df)
plt.show()

count_df = train_df[['msno', 'expiration_date', 'target']].copy()
count_df['expiration_date_ym'] = count_df['expiration_date'].apply(transform_init_time_to_ym)

exp_df = count_df[['expiration_date_ym', 'target']].groupby('expiration_date_ym').agg(['mean', 'count']).reset_index()
exp_df.columns = ['expiration_date_ym', 'replay_pb', 'play_count']

exp_df['replay_count'] = (exp_df['replay_pb'] * exp_df['play_count']).astype(np.int32)

print(exp_df.info())
plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.barplot(x='expiration_date_ym', y='play_count', data=exp_df.sort_values(by=['play_count'], ascending=False))
g.set_yscale('log', nonposy='clip')
plt.show()

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
g = sns.barplot(x='expiration_date_ym', y='replay_count', data=exp_df.sort_values(by=['replay_count'], ascending=False))
g.set_yscale('log', nonposy='clip')
plt.show()

plt.figure(figsize=(50, 15))
plt.xticks(rotation=90)
sns.barplot(x='expiration_date_ym', y='replay_pb', data=exp_df.sort_values(by=['replay_count'], ascending=False))
plt.show()

print(exp_df['replay_count'].corr(exp_df['replay_pb']))
print(exp_df['play_count'].corr(exp_df['replay_pb']))
print(exp_df['play_count'].corr(exp_df['replay_count']))

g = sns.jointplot(x="play_count", y="replay_pb", data=exp_df, kind="reg")
plt.show()
exp_df.sort_values(by=['play_count'], ascending=False)


