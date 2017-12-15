import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

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

def transform_nan_to_zero(x):
    if x == 'nan':
        return 0

    return x

def dealwith_gender(x):
    if(x=='nan'):
        return 1
    if(x=='female'):
        return 0
    if(x=='male'):
        return 2

train_df = pd.read_csv('data/train.csv')
# test_df = pd.read_csv('data/test.csv')
# comb_df = train_df.append(test_df)
members_df = pd.read_csv('data/members.csv')
# songs_df = pd.read_csv('data/songs.csv')
# song_extra_info_df = pd.read_csv('data/song_extra_info.csv')
# songs_df = songs_df.merge(song_extra_info_df, on='song_id', how='left')

members_df['registration_year'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[:4]))
members_df['registration_month'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members_df['registration_day'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[6:]))

members_df['expiration_year'] = members_df['expiration_date'].apply(lambda x: int(str(x)[:4]))
members_df['expiration_month'] = members_df['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members_df['expiration_day'] = members_df['expiration_date'].apply(lambda x: int(str(x)[6:]))

members_df['membership_days'] = members_df.apply(transform_two_dates_to_days, axis=1)
members_df.drop(['registration_init_time', 'expiration_date'], axis=1, inplace=True)

count_df = train_df[['msno', 'target']].groupby('msno').agg(['mean', 'count'])
count_df.reset_index(inplace=True)
count_df.columns = ['msno', 'replay_pb', 'play_count']
count_df['replay_count'] = (count_df['replay_pb'] * count_df['play_count']).astype(np.int32)

members_df = members_df.merge(count_df, on='msno', how='left')
members_df = members_df.drop_duplicates('msno')

members_df.fillna('nan', inplace=True)
members_df['play_count'] = members_df['play_count'].apply(transform_nan_to_zero)
members_df['replay_count'] = members_df['replay_count'].apply(transform_nan_to_zero)
members_df['replay_pb'] = members_df['replay_pb'].apply(transform_nan_to_zero)

members_df['gender'] = members_df['gender'].apply(dealwith_gender)
members_df.to_csv('data/temp_member.csv')
#归一化处理
# members_df1 = members_df[['bd','gender','registered_via','registration_year','registration_month','registration_day','expiration_year','expiration_month','expiration_day','membership_days','replay_pb','play_count','replay_count']]
# members_df1 = members_df1.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# members_df1['msno'] = members_df['msno']
# members_df1['city'] = members_df['city']
# members_df1 = members_df1[['msno','city','bd','gender','registered_via','registration_year','registration_month','registration_day','expiration_year','expiration_month','expiration_day','membership_days','replay_pb','play_count','replay_count']]
# members_df1.to_csv('data/temp_member.csv')


