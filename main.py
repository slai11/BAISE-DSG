import pdb
import csv
import numpy as np
import pandas as pd

from build_svd_features import *
from pipeline import Pipe
from model import xgboost_model, lightgbm_model
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import lightgbm as lgb

if __name__ == '__main__':
    
    train_path = 'data/archive/train_test.csv'
    val_path = 'data/archive/train_ensemble.csv'
    test_path = 'data/archive/test_clean.csv'
    
    # Set boolean to use pre-made features or build on the fly
    firsttime=True
    if firsttime:
        training_pipe = Pipe(train_path, val_path)
        train = training_pipe.make('data/pickle/train')
        train.to_csv('train_feature.csv', index=False)

        testing_pipe = Pipe(train_path, test_path)
        test = testing_pipe.make('data/pickle/test')
        test.to_csv('test_features.csv', index=False)
    else:
        train = pd.read_csv('train_feature.csv')
        test = pd.read_csv('test_features.csv')

    # Preparing train and test dataset
    featlist = train.columns.tolist()
    featlist.remove('is_listened')
    X = train[featlist].as_matrix()
    y = train['is_listened'].as_matrix()
    test_X = test[featlist].as_matrix()

    # Load model with data 
    model = lightgbm_model(X, y)
    
    # testing for submission 
    y_prob = model.predict(test_X)
    y_prob = pd.Series(y_prob, name='is_listened')
    test = pd.concat((test, y_prob), axis=1)
    
    # save to submission
    test[['sample_id', 'is_listened']].to_csv('tues_submission.csv', index=False)

