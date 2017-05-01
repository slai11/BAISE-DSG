import pandas as pd
import csv
from build_svd_features import *


class Pipe(object):

    def __init__(self, train, test, items=['media_id', 'artist_id', 'genre_id', 'album_id']):
        self.train_path = train
        self.test_path = test
        self.items = items
        self.df = pd.read_csv(test_path)
    
    def make(self):
        """uses surprise feature builder to make new features
        """
        for item in self.items:
            temp_sfb = make_sfb(item, self.train_path)
            ui_feat = temp_sfb.get_predictions(self.test_path)
            self.df.assign(ui_feat)
            self.df.drop(item)

        self.df.drop('user_id')
        return self.df

if __name__ == '__main__':
    train = 'data/train.csv'
    test = 'data/train.csv'
    pipe = Pipe(train, test, items=['genre_id'])
    df = pipe.make() 
