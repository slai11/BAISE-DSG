#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:18:08 2017

@author: joswx

This file performs preprocessing on the train and test datasets.
It adds the following features:
    1. Freshness (No. of days between ts_listen and release date)
    2. Number of user-song ratings
    3. Number of user-genre ratings
    4. Number of user-album ratings
    5. Number of user-artist ratings
"""

import csv
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime

INPUT_TRAIN_FILE_PATH      = "./data/archive/train.csv"
INPUT_TEST_FILE_PATH       = "./data/archive/test.csv"
OUTPUT_TRAIN_FILE_PATH     = "./data/archive/train_clean.csv"
OUTPUT_TEST_FILE_PATH      = "./data/archive/test_clean.csv"
OUTPUT_TRAIN_TEST_PATH     = "./data/archive/train_set.csv"
OUTPUT_VALIDATION_PATH = "./data/archive/train_validation.csv"

def _add_freshness(df):
    """Add freshness attribute to dataframe. 

    Freshness is computed by taking the difference in days between ts_listen and release date
    """
    release_date = df['release_date'].apply(lambda x: datetime.strptime(str(x), "%Y%m%d").date())
    ts_listen = df['ts_listen'].apply(lambda x: datetime.fromtimestamp(int(x)).date())
    freshness = (ts_listen - release_date).apply(lambda x: int(x.days))
    df = df.assign(freshness=freshness)
    return df

def _add_user_item_count(df, item_identifier):
    """Add user item count to dataframe. 

    User item count is number of times user-item pairs. This indicates the amount of information given to
    the SVD model.
    """
    user_item_count = df.groupby('user_id', item_identifier)['is_listened'].transform('count')
    df = df.assign(**{'user_{}'.format(item_identifier)})
    return df

def _add_features(df):
    df = _add_freshness(df)
    df = _add_user_item_count(df, 'media_id')
    df = _add_user_item_count(df, 'genre_id')
    df = _add_user_item_count(df, 'album_id')
    df = _add_user_item_count(df, 'artist_id')
    return df

def process_file(input_file_path, output_file_path):
    df = pd.read_csv(file_path)
    clean_df = add_features(df)
    clean_df.to_csv(output_file_path)
    return clean_df

def split_train_set(train_df, frac, output_train_file_path, output_validation_file_path):
    """Splits train set into training and validation set

    Arguments:
        train_df: dataframe
            train data

        frac: float
            fraction of train data

        output_train_file_path: String
            main train set will be written to this file path

        output_validation_file_path: String
            validation set will be written to this file path
    """
    train = train_df.sample(frac = frac)
    train.to_csv(OUTPUT_TRAIN_TEST_PATH)
    validation = df.drop(train.index)
    validation.to_csv(OUTPUT_VALIDATION_PATH)


if __name__ == '__main__':
    clean_train_df = process_file(INPUT_TRAIN_FILE_PATH, OUTPUT_TRAIN_FILE_PATH)
    clean_test_df = process_file(INPUT_TEST_FILE_PATH, OUTPUT_TEST_FILE_PATH)
    split_train_set(0.9)


