from surprise import SVD, BaselineOnly, GridSearch
from surprise import dataset
from surprise import evaluate, print_perf
from pathlib import Path
import pandas as pd
import time
import pickle
import os

TRAIN_FILE_PATH = "data/train.csv"
TEST_FILE_PATH = "data/test.csv"
SURPRISE_FILE_PATH = "data/{}-train_surprise_format.csv"

class SurpriseFeatureBuilder():
    def __init__(self, item_identifier='media_id', train_file_path=TRAIN_FILE_PATH, surprise_file_path=SURPRISE_FILE_PATH, 
        user_min_occurrence=20, item_min_occurrence=20):
        """SupriseFeatureBuilder formats data for ingesting and uses SVD to build a feature for a given item_identifier.

        Arguments:     
            item_identifier: String
                colname of the item 
                       
            train_file_path: string
                train file

            surprise_file_path: string
                output filtered data to this location. to be read by Surprise
                
            user_min_occurrence: int
                user must appear at least this number of times to be included

            item_min_occurrence: int
                item must appear at least this number of times to be included
        """
        self.train_file_path = train_file_path
        self.surprise_file_path = surprise_file_path
        self.item_identifier = item_identifier
        self.user_min_occurrence = user_min_occurrence
        self.item_min_occurrence = item_min_occurrence
        self.svd = SVD()

    def make_surprise_file(self, user_min_occurrence=None, item_min_occurrence=None):
        """Generates file to be ingested by Surprise.

        Arguments:                     
            user_min_occurrence: int
                user must appear at least this number of times to be included

            item_min_occurrence: int
                item must appear at least this number of times to be included
        """
        if user_min_occurrence==None:
            user_min_occurrence = self.user_min_occurrence
        if item_min_occurrence==None:
            item_min_occurrence = self.item_min_occurrence
        data = pd.read_csv(self.train_file_path)
        filtered_data = (
            data.groupby('user_id').filter(lambda x: len(x) >= user_min_occurrence)
                .groupby(self.item_identifier).filter(lambda x: len(x) >= item_min_occurrence)
                .groupby(['user_id', self.item_identifier]).mean()
            )
        print(filtered_data.shape)
        filtered_data.to_csv(path_or_buf=self.surprise_file_path, columns=['is_listened'], header=False, index=True)

    def make_file_if_missing(self):
        if not Path(self.surprise_file_path).is_file():
            print('File not found. Generating new input file')
            start_time = time.perf_counter()
            self.make_surprise_file()
            print('File generated in {}s'.format(time.perf_counter() - start_time))

    def delete_surprise_file(self):
        if Path(self.surprise_file_path).is_file():
            os.remove(self.surprise_file_path)

    def read_data(self):
        reader = dataset.Reader(line_format="user item rating", sep=',', rating_scale=(0,1), skip_lines=0)
        self.data = dataset.Dataset.load_from_file(self.surprise_file_path, reader=reader)
        self.data.split(n_folds=5)

    def eval(self):
        # Evaluate performances of our algorithm on the dataset.
        perf = evaluate(self.svd, self.data, measures=['RMSE'])
        print_perf(perf)

    def parameter_tuning(self):
        param_grid = {'n_epochs': [10, 20, 40], 'lr_all': [0.002, 0.005, 0.01],
                      'reg_all': [0.05, 0.1, 0.2]}

        print("Starting grid search...")
        start_time = time.perf_counter()
        self.grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])
        self.grid_search.evaluate(self.data)
        print('Grid search took {}s'.format(time.perf_counter() - start_time))

        self.svd = self.grid_search.best_estimator['RMSE']

        print(self.grid_search.best_score['RMSE'])
        print(self.grid_search.best_params['RMSE'])

    def train(self):
        trainset = self.data.build_full_trainset()
        self.svd.train(trainset)

    def _predict(self, user_lst, item_lst):
        assert len(user_lst) == len(item_lst)

        # gets predictions
        lst_length = len(user_lst)

        pred = [self.svd.predict(str(user_lst[idx]), str(item_lst[idx])) for idx in range(lst_length)]
        prediction, unseen = zip(*([(est, details['was_impossible']) for (_, _, _, est, details) in pred]))

        # Replace unseen with number 0, 1, 2 based on whether user, item
        unseen = [sum([self.svd.trainset.knows_user(user_lst[i]), 
                       self.svd.trainset.knows_item(item_lst[i])]) for i in range(len(user_lst))]
        return prediction, unseen

    def get_predictions(self, test_file_path):
        """Use trained model on test file

        Arguments:
            test_file_path: String 
                location of test file
        """
        data = pd.read_csv(test_file_path)
        user_lst, item_lst = data['user_id'].tolist(), data[self.item_identifier].tolist()
        predictions, unseen = self._predict(user_lst, item_lst)
        return {"{}_svd".format(self.item_identifier): predictions,
                "{}_unseen".format(self.item_identifier): unseen}

def make_sfb(item_identifier, train_file_path, model_name="", user_min_occurrence=20, item_min_occurrence=20):
    """Make a SurpriseFeatureBuilder trained on the given file

    Arguments:
        item_identifier: String
            colname of the item 

        train_file_path: String 
            location of train file

        user_min_occurrence: int
            user must appear at least this number of times to be included

        item_min_occurrence: int
            item must appear at least this number of times to be included

    Returns:
        sfb: SurpriseFeatureBuilder object
            sfb trained on train file
    """
    model_pickle_name = "{}_{}.p".format(model_name, item_identifier)
    try:
        sfb = pickle.load(open(model_pickle_name, "rb" ))
    except FileNotFoundError:
        sfb = SurpriseFeatureBuilder(item_identifier, train_file_path, 
            SURPRISE_FILE_PATH.format(item_identifier), 
            user_min_occurrence, item_min_occurrence)
        sfb.make_file_if_missing()
        sfb.read_data()
        sfb.train()
        sfb.delete_surprise_file()
        pickle.dump(sfb, open(model_pickle_name, "wb"))
    return sfb

if __name__ == '__main__':
    media_sfb = make_sfb('genre_id', TRAIN_FILE_PATH, user_min_occurrence=3000, item_min_occurrence=20)
    d = media_sfb.get_predictions(TEST_FILE_PATH)

