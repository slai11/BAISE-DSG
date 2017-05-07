from surprise import SVD, BaselineOnly, GridSearch
from surprise import dataset
from surprise import evaluate, print_perf
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import time
import random
import pickle
import os

"""

How to use?

1. To get a trained SFB model, use the make_sfb function.
2. To get predictions using this model, use the get_predictions method.
3. To get latent features, use the get_latent_features method.

For example,

> my_sfb = make_sfb('media_id', TRAIN_FILE_PATH, user_min_occurrence=20, item_min_occurrence=20, number_of_folds=1)
> my_sfb.get_predictions(TRAIN_FILE_PATH)
> user_baseline, item_baseline, user_vectors. item_vectors = my_sfb.get_latent_features()[0]

"""

TRAIN_FILE_PATH = "data/archive/train.csv"
TEST_FILE_PATH = "data/archive/test.csv"
TEMP_FILE_PATH = "data/archive/{}-train_surprise_format.csv"

def make_sfb(item_identifier, train_file_path, model_name="", user_min_occurrence=20, item_min_occurrence=20, no_of_folds=5):
    """Make a SurpriseFeatureBuilder trained on the given file.

    Arguments:
        item_identifier: String
            colname of the item 

        train_file_path: String 
            location of train file

        user_min_occurrence: int
            user must appear at least this number of times to be included

        item_min_occurrence: int
            item must appear at least this number of times to be included

        no_of_folds: int
            number of folds that the data is split for training

    Returns:
        sfb: SurpriseFeatureBuilder object
            sfb trained on train file
    """
    model_pickle_name = "pickles/{}_{}.p".format(model_name, item_identifier)
    try:
        sfb = pickle.load(open(model_pickle_name, "rb" ))
    except FileNotFoundError:
        sfb = SurpriseFeatureBuilder(item_identifier, train_file_path, 
            TEMP_FILE_PATH.format(item_identifier), 
            user_min_occurrence, item_min_occurrence, no_of_folds)
        sfb.delete_surprise_file()
        pickle.dump(sfb, open(model_pickle_name, "wb"))
    return sfb

def mini_train(pair):
    svd, sub_trainset = pair
    svd.train(sub_trainset)
    return svd

class SurpriseFeatureBuilder():
    def __init__(self, item_identifier='media_id', train_file_path=TRAIN_FILE_PATH, temp_file_path=TEMP_FILE_PATH, 
        user_min_occurrence=20, item_min_occurrence=20, no_of_folds=5):
        """SupriseFeatureBuilder formats data for ingesting and uses SVD to build a feature for a given item_identifier.

        Arguments:     
            item_identifier: String
                colname of the item 
                       
            train_file_path: string
                train file

            temp_file_path: string
                output filtered data to this location. to be read by Surprise
                
            user_min_occurrence: int
                user must appear at least this number of times to be included

            item_min_occurrence: int
                item must appear at least this number of times to be included

            no_of_folds: int
                number of folds that the data is split for training

        Other Attributes:
            main_svd: SVD object
                SVD trained on full dataset

            sub_svds: list of SVD objects 
                SVDs trained on subsets of the full dataset

        """
        if no_of_folds < 1:
            raise ValueError("number of folds must be at least 1")

        self.item_identifier = item_identifier
        self.train_file_path = train_file_path
        self.temp_file_path = temp_file_path
        self.user_min_occurrence = user_min_occurrence
        self.item_min_occurrence = item_min_occurrence
        self.no_of_folds = no_of_folds

        self.main_svd = SVD()
        self.sub_svds = [SVD() for _ in range(no_of_folds)]
        self.is_trained = False

        self.make_file_if_missing()
        self.read_data()
        self.train()
        print("SFB initialized")


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
        filtered_data.to_csv(path_or_buf=self.temp_file_path, columns=['is_listened'], header=False, index=True)

    def make_file_if_missing(self):
        if not Path(self.temp_file_path).is_file():
            print('File not found. Generating new input file')
            start_time = time.perf_counter()
            self.make_surprise_file()
            print('File generated in {}s'.format(time.perf_counter() - start_time))

    def delete_surprise_file(self):
        if Path(self.temp_file_path).is_file():
            os.remove(self.temp_file_path)

    def read_data(self):
        """Read data from temp file in surprise format and perform k fold splitting on data"""
        reader = dataset.Reader(line_format="user item rating", sep=',', rating_scale=(0,1), skip_lines=0)

        self.main_dataset = dataset.Dataset.load_from_file(self.temp_file_path, reader=reader)
        self.sub_datasets = [dataset.Dataset.load_from_file(self.temp_file_path, reader=reader) for _ in range(self.no_of_folds)]
        
        ratings = self.sub_datasets[0].raw_ratings

        if self.no_of_folds > 1:
            self.ratings_exclude_size = len(ratings)//self.no_of_folds
        else:
            self.ratings_exclude_size = 0

        for idx, data in enumerate(self.sub_datasets):
            data.raw_ratings = [ele for idx, ele in enumerate(data.raw_ratings) if idx not in range(idx*self.ratings_exclude_size, (idx+1)*self.ratings_exclude_size)]

    def train(self):
        main_trainset = self.main_dataset.build_full_trainset()
        sub_trainsets = [d.build_full_trainset() for d in self.sub_datasets]

        self.main_svd.train(main_trainset)
        with Pool(4) as p:
            self.sub_svds = p.map(mini_train, [[self.sub_svds[i], sub_trainsets[i]] for i in range(self.no_of_folds)])

        self.is_trained = True

    def _predict(self, svd, user_lst, item_lst):

        assert len(user_lst) == len(item_lst)

        # gets predictions
        lst_length = len(user_lst)

        pred = [svd.predict(str(user_lst[idx]), str(item_lst[idx])) for idx in range(lst_length)]
        prediction, unseen = zip(*([(est, details['was_impossible']) for (_, _, _, est, details) in pred]))

        # Replace unseen with number 0, 1, 2 based on whether user, item
        unseen = [sum([svd.trainset.knows_user(user_lst[i]), 
                       svd.trainset.knows_item(item_lst[i])]) for i in range(len(user_lst))]
        return prediction, unseen

    def get_predictions_on_same_dataset(self):

        predictions = []
        unseens = []
        ratings = self.main_dataset.raw_ratings
        for idx, svd in enumerate(self.sub_svds):
            test_data = [ele for idx, ele in enumerate(ratings) if idx in range(idx*self.ratings_exclude_size, (idx+1)*self.ratings_exclude_size)]
            user_lst, item_lst, score_lst, _ = zip(*test_data)
            prediction, unseen = self._predict(svd, user_lst, item_lst)

            predictions.extend(prediction)
            unseens.extend(unseen)

        return {"{}_svd".format(self.item_identifier): predictions,
                "{}_unseen".format(self.item_identifier): unseen}

    def get_predictions(self, test_file_path):
        """Use trained model on test file

        Arguments:
            test_file_path: String 
                location of test file
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained")

        if test_file_path == self.train_file_path:
            return self.get_predictions_on_same_dataset()

        data = pd.read_csv(test_file_path)
        user_lst, item_lst = data['user_id'].tolist(), data[self.item_identifier].tolist()

        predictions, unseen = self._predict(self.main_svd, user_lst, item_lst)
        return {"{}_svd".format(self.item_identifier): predictions,
                "{}_unseen".format(self.item_identifier): unseen}

    def get_latent_features(self):
        """Returns calculated baselines, and user and item vectors"""
        return [(s.bu, s.bi, s.pu, s.qi) for s in self.sub_svds]

if __name__ == '__main__':
    media_sfb = make_sfb('media_id', TRAIN_FILE_PATH, user_min_occurrence=20, item_min_occurrence=20, no_of_folds=4)
    media_sfb.get_predictions(TRAIN_FILE_PATH)
    media_sfb.get_predictions(TEST_FILE_PATH)
    user_baseline, item_baseline, user_vectors, item_vectors = media_sfb.get_latent_features()[0]
    print(user_vectors)
    print(item_vectors)


