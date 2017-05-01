import pandas as pd
import csv
from build_svd_features import *
import pdb

class Pipe(object):

    def __init__(self, train, test, items=['media_id', 'artist_id', 'genre_id', 'album_id']):
        self.train_path = train
        self.test_path = test
        self.items = items
        self.df = pd.read_csv(self.test_path)
    
    def make(self, model_name):
        """uses surprise feature builder to make new features
        """
        for item in self.items:
            print("Building features for {} now.".format(item))
            temp_sfb = make_sfb(item, self.train_path, model_name)
            ui_feat = temp_sfb.get_predictions(self.test_path)
            self.df = self.df.assign(**ui_feat)
            self.df.drop(item, axis=1, inplace=True)
            
        self.df.drop('user_id', axis=1, inplace=True)
        return self.df

if __name__ == '__main__':
    train = 'data/train.csv'
    test = 'data/train.csv'
    pipe = Pipe(train, test, items=['genre_id'])
    df = pipe.make() 
