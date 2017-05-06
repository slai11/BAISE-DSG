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
    def convert_to_date(x):
        try:
            return datetime.fromtimestamp(int(x)).date()
        except OSError as e:
            return

    def difference_to_days(x):
        try:
            return int(x.days)
        except:
            return
        
    release_date = df['release_date'].apply(lambda x: datetime.strptime(str(x), "%Y%m%d").date())
    ts_listen = df['ts_listen'].apply(convert_to_date)
    freshness = (ts_listen - release_date).apply(difference_to_days)
    df = df.assign(freshness=freshness)
    return df

def _get_user_item_count(df, item_identifier):
    """Add user item count to dataframe. 

    User item count is number of times user-item pairs. This indicates the amount of information given to
    the SVD model.
    """
    user_item_count = df.groupby(['user_id', item_identifier])['user_id'].count()
    user_item_count.name = '{} count'.format(item_identifier)
    return user_item_count

def _add_user_item_count(df, user_item_count, item_identifier):
    df = df.join(user_item_count, on=['user_id', 'media_id'])
    return df

def process_files(input_train_file_path, output_train_file_path, input_test_file_path, output_test_file_path):
    """Reads files, adds features, then outputs data"""
    
    train_df = pd.read_csv(input_train_file_path)
    test_df = pd.read_csv(input_train_file_path)

    for item_identifier in ['media_id', 'genre_id', 'album_id', 'artist_id']:
        user_item_count = _get_user_item_count(train_df, item_identifier)
        train_df = _add_user_item_count(train_df, user_item_count, item_identifier)
        test_df = _add_user_item_count(test_df, user_item_count, item_identifier)

    train_df = _add_freshness(train_df)
    test_df = _add_freshness(test_df)

    train_df.to_csv(output_train_file_path)
    test_df.to_csv(output_test_file_path)
    print("File at {} cleaned and written to {}".format(input_train_file_path, output_train_file_path))
    print("File at {} cleaned and written to {}".format(input_test_file_path, output_test_file_path))
    return train_df, test_df

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
    train.to_csv(output_train_file_path)
    validation = train.drop(train.index)
    validation.to_csv(output_validation_file_path)


if __name__ == '__main__':
    clean_train_df, clean_test_df = process_files(INPUT_TRAIN_FILE_PATH, OUTPUT_TRAIN_FILE_PATH, INPUT_TEST_FILE_PATH, OUTPUT_TEST_FILE_PATH)
    split_train_set(clean_train_df, 0.9, OUTPUT_TRAIN_TEST_PATH, OUTPUT_VALIDATION_PATH)


