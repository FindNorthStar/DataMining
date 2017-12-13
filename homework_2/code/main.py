import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime as dt
import random

import json
import os.path
import gc

MODEL_FILE_NAME = 'model.txt'



train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
comb_df = train_df.append(test_df)
songs_df = pd.read_csv('data/songs.csv')
song_extra_info_df = pd.read_csv('data/song_extra_info.csv')
members_df = pd.read_csv('data/members.csv')

def custom_cv(params, train_set, hold_out_set=None, k_fold=5, num_boost_round=20):
    x_train = train_set.data
    y_train = train_set.label
    n = x_train.shape[0]
    unit = n // k_fold

    cv_scores = []
    for k in range(k_fold):
        ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if k < k_fold - 1:
            continue
        ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x_cv_valid = None
        y_cv_valid = None
        if k == k_fold - 1:
            x_cv_valid = x_train[unit * k:]
            y_cv_valid = y_train[unit * k:]
        else:
            x_cv_valid = x_train[unit * k: unit * (k + 1)]
            y_cv_valid = y_train[unit * k: unit * (k + 1)]

        x_cv_train = None
        y_cv_train = None
        if k == 0:
            x_cv_train = x_train[unit * (k + 1):]
            y_cv_train = y_train[unit * (k + 1):]
        elif k == k_fold - 1:
            x_cv_train = x_train[:unit * k]
            y_cv_train = y_train[:unit * k]
        else:
            x_cv_train = x_train[:unit * k].append(x_train[unit * (k + 1):])
            y_cv_train = y_train[:unit * k].append(y_train[unit * (k + 1):])

        cv_train_set = lgb.Dataset(x_cv_train, y_cv_train)
        cv_valid_set = lgb.Dataset(x_cv_valid, y_cv_valid)
        watchlist = [cv_valid_set]

        # not tested yet
        if hold_out_set is not None:
            watchlist.append(hold_out_set)
        model = lgb.train(params, train_set=cv_train_set, valid_sets=watchlist,
                          num_boost_round=num_boost_round, verbose_eval=5)
        print(model.best_score)
        cv_scores.append(model.best_score['valid_1']['auc'])

    tip_txt = '[CV]'
    tip_txt += ' ' + str(cv_scores)
    mean_cv_score = np.mean(cv_scores)
    tip_txt += '{ auc score=' + str(mean_cv_score) + ' }'
    print(tip_txt)

    return mean_cv_score


def custom_grid_search(params, own_grid_params, train_set, valid_set, num_boost_round=20):
    keys = []
    values = [list()]
    for key, value in own_grid_params.items():
        keys.append(key)
        new_values = []
        for item in values:
            for val in value:
                new_values.append(item + [val])
        values = new_values

    watchlist = [valid_set]
    grid_best_params = None
    grid_best_score = None

    for comb in values:
        own_params = {}
        for idx in range(len(keys)):
            own_params[keys[idx]] = comb[idx]
            params[keys[idx]] = comb[idx]

        cv_score = custom_cv(params, train_set, valid_set, k_fold=4, num_boost_round=num_boost_round)

        tip_txt = '[GridSearch]'
        for idx in range(len(keys)):
            tip_txt += ' ' + str(keys[idx]) + '=' + str(comb[idx])
        tip_txt += ' { best_score: ' + str(cv_score) + ' }'
        print(tip_txt)

        if grid_best_score is None or cv_score > grid_best_score:
            grid_best_params, grid_best_score = own_params, cv_score

    tip_txt = '[GS Best Result]'
    for key, val in grid_best_params.items():
        tip_txt += ' ' + str(key) + '=' + str(val)
    tip_txt += ' { best_score: ' + str(grid_best_score) + ' }'
    print(tip_txt)

    return grid_best_params


def one_hot_encode_system_tab(x):
    return 1 if x == 'my library' else 0


def one_hot_encode_screen_name(x):
    return 1 if x == 'Local playlist more' or x == 'My library' else 0


def one_hot_encode_source_type(x):
    return 1 if x == 'local-library' or x == 'local-playlist' else 0


def one_hot_encode_source(x):
    return 1 if x >= 0.6 else 0



train_df['source_system_tab'].fillna('others', inplace=True)
test_df['source_system_tab'].fillna('others', inplace=True)

train_df['source_screen_name'].fillna('others', inplace=True)
test_df['source_screen_name'].fillna('others', inplace=True)

train_df['source_type'].fillna('nan', inplace=True)
test_df['source_type'].fillna('nan', inplace=True)

assert(~train_df.isnull().any().any())
assert(~test_df.isnull().any().any())

train_df['source_merged'] = train_df['source_system_tab'].map(str) + ' | ' + train_df['source_screen_name'].map(str) + ' | ' + train_df['source_type'].map(str)
test_df['source_merged'] = test_df['source_system_tab'].map(str) + ' | ' + test_df['source_screen_name'].map(str) + ' | ' + test_df['source_type'].map(str)

count_df = train_df[['source_merged', 'target']].groupby('source_merged').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['source_merged', 'source_replay_pb', 'source_replay_count']

train_df = train_df.merge(count_df, on='source_merged', how='left')
test_df = test_df.merge(count_df, on='source_merged', how='left')

train_df['1h_source'] = train_df['source_replay_pb'].apply(one_hot_encode_source)
test_df['1h_source'] = test_df['source_replay_pb'].apply(one_hot_encode_source)

train_df.drop(['source_merged', 'source_replay_pb', 'source_replay_count'], axis=1, inplace=True)
test_df.drop(['source_merged', 'source_replay_pb', 'source_replay_count'], axis=1, inplace=True)

train_df['1h_system_tab'] = train_df['source_system_tab'].apply(one_hot_encode_system_tab)
train_df['1h_screen_name'] = train_df['source_screen_name'].apply(one_hot_encode_screen_name)
train_df['1h_source_type'] = train_df['source_type'].apply(one_hot_encode_source_type)

test_df['1h_system_tab'] = test_df['source_system_tab'].apply(one_hot_encode_system_tab)
test_df['1h_screen_name'] = test_df['source_screen_name'].apply(one_hot_encode_screen_name)
test_df['1h_source_type'] = test_df['source_type'].apply(one_hot_encode_source_type)

# never drop, important
# train_df.drop(["source_system_tab", "source_screen_name", "source_type"], axis=1, inplace=True)
# test_df.drop(["source_system_tab", "source_screen_name", "source_type"], axis=1, inplace=True)

def parse_str_to_date(date_str):
    # [format] yyyymmdd
    date_str = str(date_str)
    assert (isinstance(date_str, str))
    assert (len(date_str) == 8)

    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:])
    return dt.date(year, month, day)


def transform_two_dates_to_days(row):
    start = parse_str_to_date(row['registration_init_time'])
    end = parse_str_to_date(row['expiration_date'])
    delta = end - start
    return delta.days


def transform_bd_outliers(bd):
    # figure is from "exploration"
    if bd >= 120 or bd <= 7:
        return 'nan'
    mean = 28.99737187910644
    std = 9.538470787507382
    return bd if abs(bd - mean) <= 3 * std else 'nan'


def transform_outliers(x, mean, std):
    return x if np.abs(x - mean) <= 3 * std else -1


def one_hot_encode_via(x):
    return 0 if x == 4 else 1


def transform_init_time_to_ym(time):
    time_str = str(time)
    year = int(time_str[:4])
    month = int(time_str[4:6])
    return int("%04d%02d" % (year, month))

    # def custom_gender_random_seed(x):
    #    if x is not np.nan:
    #        return x
    #    return random.choice(['female', 'male'])

members_df['membership_days'] = members_df.apply(transform_two_dates_to_days, axis=1)

members_df['registration_init_year'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[:4]))
members_df['registration_init_month'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[4:6]))

members_df['expiration_date_year'] = members_df['expiration_date'].apply(lambda x: int(str(x)[:4]))
members_df['expiration_date_month'] = members_df['expiration_date'].apply(lambda x: int(str(x)[4:6]))

members_df.drop(['registration_init_time'], axis=1, inplace=True)

members_df['bd'] = members_df['bd'].apply(transform_bd_outliers)

members_df['gender'].fillna('nan', inplace=True)

members_df['1h_via'] = members_df['registered_via'].apply(one_hot_encode_via)

assert(~members_df.isnull().any().any())
members_df.head(15)


# reference http://isrc.ifpi.org/en/isrc-standard/code-syntax
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


def transform_isrc_to_reg(isrc):
    if type(isrc) != str:
        return np.nan
    registration = isrc[2:5]

    return registration


def transfrom_isrc_to_desig(isrc):
    if type(isrc) != str:
        return np.nan
    designation = isrc[7:]

    return designation


def one_hot_encode_year(x):
    return 1 if 2013 <= float(x) <= 2017 else 0


def one_hot_encode_country(x):
    return 1 if x == 'TW' or x == 'CN' or x == 'HK' else 0

song_extra_info_df['song_year'] = song_extra_info_df['isrc'].apply(transform_isrc_to_year)
# song_extra_info_df['song_country'] = song_extra_info_df['isrc'].apply(transform_isrc_to_country)
# song_extra_info_df['song_registration'] = song_extra_info_df['isrc'].apply(transform_isrc_to_reg)
# song_extra_info_df['song_designation'] = song_extra_info_df['isrc'].apply(transfrom_isrc_to_desig)

song_extra_info_df['1h_song_year'] = song_extra_info_df['song_year'].apply(one_hot_encode_year)
# song_extra_info_df['1h_song_country'] = song_extra_info_df['song_country'].apply(one_hot_encode_country)

song_extra_info_df.drop(['isrc', 'name'], axis=1, inplace=True)

song_extra_info_df['song_year'].fillna(2017, inplace=True)
# song_extra_info_df['song_registration'].fillna('***', inplace=True)

assert(~song_extra_info_df.isnull().any().any())
song_extra_info_df.head(15)


def parse_splitted_category_to_number(x):
    if x is np.nan:
        return 0

    x = str(x)
    x.replace('/', '|')
    x.replace(';', '|')
    x.replace('\\', '|')
    x.replace(' and ', '|')
    x.replace('&', '|')
    x.replace('+', '|')
    return x.count('|') + 1


def one_hot_encode_lang(x):
    return 1 if x in [-1, 17, 45] else 0

songs_df['genre_count'] = songs_df['genre_ids'].apply(parse_splitted_category_to_number)
songs_df['composer_count'] = songs_df['composer'].apply(parse_splitted_category_to_number)
songs_df['lyricist_count'] = songs_df['lyricist'].apply(parse_splitted_category_to_number)

songs_df['1h_lang'] = songs_df['language'].apply(one_hot_encode_lang)

songs_df['1h_song_length'] = songs_df['song_length'].apply(lambda x: 1 if x <= 239738 else 0)

songs_df['language'].fillna('nan', inplace=True)
songs_df['composer'].fillna('nan', inplace=True)
songs_df['lyricist'].fillna('nan', inplace=True)
songs_df['genre_ids'].fillna('nan', inplace=True)
# songs_df.drop(['language'], axis=1, inplace=True)
assert(~songs_df.isnull().any().any())
songs_df.head(15)

count_df = train_df['song_id'].value_counts().reset_index()
count_df.columns = ['song_id', 'play_count']

train_df = train_df.merge(count_df, on='song_id', how='left')
train_df['play_count'].fillna(0, inplace=True)

count_df = comb_df['song_id'].value_counts().reset_index()
count_df.columns = ['song_id', 'play_count']

test_df = test_df.merge(count_df, on='song_id', how='left')
test_df['play_count'].fillna(0, inplace=True)

songs_df = songs_df.merge(song_extra_info_df, on='song_id', how='left')

train_df = train_df.merge(songs_df, on='song_id', how='left')
test_df = test_df.merge(songs_df, on='song_id', how='left')

train_df = train_df.merge(members_df, on='msno', how='left')
test_df = test_df.merge(members_df, on='msno', how='left')

train_df.info()
test_df.info()

for column in train_df.columns:
    if train_df[column].dtype == object:
        train_df[column] = train_df[column].astype('category')
for column in test_df.columns:
    if test_df[column].dtype == object:
        test_df[column] = test_df[column].astype('category')

x = train_df.drop(['target'], axis=1)
y = train_df['target']

# take appropriate data to train
# x = train_df.drop(['target'], axis=1).head(train_df.shape[0] - test_df.shape[0])
# y = train_df['target'].head(train_df.shape[0] - test_df.shape[0]).values

# take the last # rows of train_df as valid set where # means number of rows in test_df
x_valid = train_df.drop(['target'], axis=1).tail(test_df.shape[0])
y_valid = train_df['target'].tail(test_df.shape[0])

x_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']

train_df.head(15)


corr = train_df.corr()

f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()

train_set = lgb.Dataset(x, y)
valid_set = lgb.Dataset(x_valid, y_valid, free_raw_data=False)
watchlist = [valid_set]

params = dict({
    'learning_rate': 0.2,
    'application': 'binary',
    'min_data_in_leaf': 10,
#    'max_depth': 10,
    'num_leaves': 2 ** 7,
    'max_bin': 255,
    'verbosity': 0,
    'metric': 'auc'
})

grid_params = {
    'learning_rate': [0.1, 0.2],
    'max_depth': [8, 10],
}

# best_grid_params = custom_grid_search(params, grid_params, train_set, hold_out_set, num_boost_round=20)
# for key, val in best_grid_params.items():
#     params[key] = best_grid_params[key]

cv_score = custom_cv(params, train_set, valid_set, k_fold=4, num_boost_round=100)
print(cv_score)

model = lgb.train(params, train_set=train_set, valid_sets=watchlist, num_boost_round=100, verbose_eval=5)
y_test = model.predict(x_test)

plot_df = pd.DataFrame({'features': train_df.columns[train_df.columns != 'target'],
                        'importances': model.feature_importance()})
plot_df = plot_df.sort_values('importances', ascending=False)

sns.barplot(x = plot_df.importances, y = plot_df.features)
plt.show()

submission_df = pd.DataFrame()
submission_df['id'] = test_ids
submission_df['target'] = y_test
# string file compression reduces file size
submission_df.to_csv('data/submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
submission_df.info()