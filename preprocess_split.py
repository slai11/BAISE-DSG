#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:18:08 2017

@author: zhouweixin
"""

import csv
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime

INPUT_FILE_PATH = "./data/train.csv"
OUTPUT_CLEAN_FILE_PATH = "./data/total_clean.csv"
OUTPUT_TRAIN_TEST_PATH = "./data/train_test.csv"
OUTPUT_TRAIN_EMSEMBLE_PATH = "./data/train_ensemble.csv"


if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE_PATH)
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

    #split
    df.to_csv(OUTPUT_CLEAN_FILE_PATH)
    train = df.sample(frac = 0.5)
    train.to_csv(OUTPUT_TRAIN_TEST_PATH)
    test = df.drop(train.index)
    test.to_csv(OUTPUT_TRAIN_EMSEMBLE_PATH)


