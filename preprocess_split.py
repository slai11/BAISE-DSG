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

INPUT_TRAIN_FILE_PATH = "./data/archive/train.csv"
INPUT_TEST_FILE_PATH = "./data//archive/test.csv"
OUTPUT_CLEAN_FILE_PATH = "./data/archive/train_clean.csv"
OUTPUT_TEST_FILE_PATH = "./data/archive/test_clean.csv"
OUTPUT_TRAIN_TEST_PATH = "./data/archive/train_test.csv"
OUTPUT_TRAIN_EMSEMBLE_PATH = "./data/archive/train_ensemble.csv"


def add_fresh(df):
    freshness= []
    for nrow in range (len(df.index)):
        #release_date
        td1 = df["release_date"][nrow]
        td1 = datetime.strptime(str(td1), "%Y%m%d").date()
        #ts_listen
        td2 = datetime.fromtimestamp(int(df["ts_listen"][nrow])).date()
        curr = int((td2 - td1).days)
        freshness.append(curr)
        print(nrow)
    df['freshness'] = freshness
    return df

if __name__ == '__main__':
    train_df = pd.read_csv(INPUT_TRAIN_FILE_PATH)
    test_df = pd.read_csv(INPUT_TEST_FILE_PATH)
    train_output = add_fresh(train_df)
    test_output = add_fresh(test_df)

    #split
    train_output.to_csv(OUTPUT_CLEAN_FILE_PATH)
    test_output.to_csv(OUTPUT_TEST_FILE_PATH )
    train = train_output.sample(frac = 0.9)
    train.to_csv(OUTPUT_TRAIN_TEST_PATH)
    test = train_output.drop(train.index)
    test.to_csv(OUTPUT_TRAIN_EMSEMBLE_PATH)


