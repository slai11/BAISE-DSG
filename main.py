import pdb
import csv
import numpy as np
import pandas as pd

from pipeline import Pipe
from model import lightgbm_model
from datetime import datetime
import lightgbm as lgb

if __name__ == '__main__':
    
    # Input files
    train_path = 'data/archive/train_clean.csv'
    test_path = 'data/archive/test_clean.csv'

    # Intermediate files
    TRAIN_PATH_INTERMEDIATE = 'data/archive/train_intermediate.csv'
    TEST_PATH_INTERMEDIATE = 'data/archive/test_intermediate.csv'


    # Set boolean to use pre-made features or build on the fly
    firsttime=True
    if firsttime:
        training_pipe = Pipe(train_path, train_path)
        train = training_pipe.make('train_intermediate')
        train.to_csv(TRAIN_PATH_INTERMEDIATE, index=False)

        testing_pipe = Pipe(train_path, test_path)
        test = testing_pipe.make('test_intermediate')
        test.to_csv(TEST_PATH_INTERMEDIATE, index=False)
    else:
        train = pd.read_csv('train_feature.csv')
        test = pd.read_csv('test_features.csv')

    # Preparing train and test dataset for ensemble layer
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
    test[['sample_id', 'is_listened']].to_csv('submission.csv', index=False)

