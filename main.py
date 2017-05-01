import pdb
import csv
import numpy as np
import pandas as pd

from build_svd_features import *
from pipeline import Pipe
from model import xgboost_model

import xgboost as xgb

if __name__ == '__main__':
    train_path, vali_path = josephmethodi()

    train_path = 'data/train.csv'
    val_path = train_path
    test_path = 'data/test.csv'
    
    training_pipe = Pipe(train_path, val_path)
    train = training_pipe.make()
    pdb.set_trace()
    testing_pipe = Pipe(test_path, test_path)
    test = testing_pipe.make()
    
    model = xgboost_model()
    y_prob = model.predict()

