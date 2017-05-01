import pdb
import csv
import numpy as np
import pandas as pd

from build_svd_features import *
from pipeline import Pipe
from model import xgboost_model

import xgboost as xgb

if __name__ == '__main__':
    #train_path, vali_path = josephmethodi()

    train_path = 'data/train.csv'
    val_path = train_path
    test_path = 'data/test.csv'
    
    training_pipe = Pipe(train_path, val_path)
    train = training_pipe.make('train')
    featlist = train.columns.tolist().remove('is_listened')
    train_X = train[featlist].as_matrix()
    train_y = train['is_listened'].as_matrix()
    train =  xgb.XMatrix(train_X, train_y, missing=-999)

    pdb.set_trace()
    testing_pipe = Pipe(train_path, test_path)
    test = testing_pipe.make('test')
    test_X = test[featlist].as_matrix()

    pdb.set_trace()
    
    model = xgboost_model()
    
    test = xgb.XMatrix(test_X, missing=-999)
    
    y_prob = model.predict()

