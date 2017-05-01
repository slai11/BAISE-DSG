#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:18:08 2017

@author: joswx
"""

import csv
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime

INPUT_TRAIN_FILE_PATH = "./data/train.csv"
INPUT_TEST_FILE_PATH = "./data/test.csv"
OUTPUT_CLEAN_FILE_PATH = "./data/train_clean.csv"
OUTPUT_TEST_FILE_PATH = "./data/test_clean.csv"
OUTPUT_TRAIN_TEST_PATH = "./data/train_test.csv"
OUTPUT_TRAIN_EMSEMBLE_PATH = "./data/train_ensemble.csv"

def add_fresh(df):
    """Add freshness attribute to dataframe. 

    Freshness is computed by taking the difference in days between ts_listen and release date
    """
    release_date = train_df['release_date'].apply(lambda x: datetime.strptime(str(x), "%Y%m%d").date())
    ts_listen = train_df['ts_listen'].apply(lambda x: datetime.fromtimestamp(int(x)).date())
    freshness = (ts_listen - release_date).apply(lambda x: int(x.days))
    df = df.assign(freshness=freshness)
    return df

if __name__ == '__main__':
    train_df = pd.read_csv(INPUT_TRAIN_FILE_PATH)
    test_df = pd.read_csv(INPUT_TEST_FILE_PATH)
    train_output = add_fresh(train_df)
    test_output = add_fresh(test_df)

    #split
    train_output.to_csv(OUTPUT_CLEAN_FILE_PATH)
    test_output.to_csv(OUTPUT_TEST_FILE_PATH )
    train = train_output.sample(frac = 0.5)
    train.to_csv(OUTPUT_TRAIN_TEST_PATH)
    test = df.drop(train.index)
    test.to_csv(OUTPUT_TRAIN_EMSEMBLE_PATH)


