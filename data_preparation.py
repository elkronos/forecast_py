import numpy as np
import pandas as pd

def generate_data():
    date_rng = pd.date_range(start='1/1/2020', end='1/10/2023', freq='D')
    df = pd.DataFrame(date_rng, columns=['date'])
    df['data'] = np.random.randn(len(date_rng))
    df.set_index('date', inplace=True)
    return df

def train_test_split(df):
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]
    return train, test

def create_features(data_frame):
    data_frame['year'] = data_frame.index.year
    data_frame['month'] = data_frame.index.month
    data_frame['day'] = data_frame.index.day
    data_frame['day_of_week'] = data_frame.index.dayofweek
    data_frame['day_of_year'] = data_frame.index.dayofyear
    data_frame['week_of_year'] = data_frame.index.isocalendar().week
    data_frame['quarter'] = data_frame.index.quarter
    data_frame['lag_1'] = data_frame['data'].shift(1)
    data_frame['lag_7'] = data_frame['data'].shift(7)
    data_frame['rolling_mean_7'] = data_frame['data'].rolling(window=7).mean()
    data_frame['rolling_std_7'] = data_frame['data'].rolling(window=7).std()
    return data_frame
