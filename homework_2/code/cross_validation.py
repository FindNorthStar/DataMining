# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime as dt

train_df = pd.read_csv('../../KKBox/train.csv')
test_df = pd.read_csv('../../KKBox/test.csv')
comb_df = train_df.append(test_df)
songs_df = pd.read_csv('../../KKBox/songs.csv')
song_extra_info_df = pd.read_csv('../../KKBox/song_extra_info.csv')
members_df = pd.read_csv('../../KKBox/members.csv')

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


if __name__ == '__main__':
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

    x = train_df.drop(['target'], axis=1)
    y = train_df['target']

    # take the last # rows of train_df as valid set where # means number of rows in test_df
    x_valid = train_df.drop(['target'], axis=1).tail(test_df.shape[0])
    y_valid = train_df['target'].tail(test_df.shape[0])

    train_set = lgb.Dataset(x, y)
    valid_set = lgb.Dataset(x_valid, y_valid, free_raw_data=False)

    cv_score = custom_cv(params, train_set, valid_set, k_fold=4, num_boost_round=100)
    print(cv_score)