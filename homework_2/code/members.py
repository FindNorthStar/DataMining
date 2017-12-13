import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime as dt
import random

import gc

MODEL_FILE_NAME = 'model.txt'

members_df = pd.read_csv('data/members.csv')
members_df.info()
members_df.head(15)

members_df['registration_year'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[:4]))
members_df['registration_month'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members_df['registration_day'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[6:]))

members_df['expiration_year'] = members_df['expiration_date'].apply(lambda x: int(str(x)[:4]))
members_df['expiration_month'] = members_df['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members_df['expiration_day'] = members_df['expiration_date'].apply(lambda x: int(str(x)[6:]))

# members_df.drop(['registration_init_time', 'expiration_date'], axis=1, inplace=True)
members_df.head(15)


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


members_df['membership_days'] = members_df.apply(transform_two_dates_to_days, axis=1)
members_df.drop(['registration_init_time', 'expiration_date'], axis=1, inplace=True)
print(members_df.head(15))

