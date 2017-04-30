from surprise import SVD, BaselineOnly, GridSearch
from surprise import dataset
from surprise import evaluate, print_perf
from pathlib import Path
import pandas as pd
import time
import pickle

TRAIN_FILE_PATH = "data/train.csv"
TEST_FILE_PATH = "data/test.csv"
SURPRISE_FILE_PATH = "data/train_surprise_format.csv"
PREDICTIONS_FILE_PATH = "data/test_predict.csv"

class SurpriseFeatureBuilder():
	def __init__(self, train_file_path, test_file_path, surprise_file_path, predictions_file_path, item_identifier='media_id'):
		"""SupriseFeatureBuilder formats data for ingesting and uses SVD to build a feature for a given item_identifier.

		Attributes:
			train_file_path: string
				train file

			test_file_path: string
				test file

			surprise_file_path: string
				output filtered data to this location. to be read by Surprise

			predictions_file_path: string
				output location to store predictions
				
	        item_identifier: string {'media_id', 'genre_id', 'artist_id', 'album_id'}
	            column names from df to specify which item to match users against 
	    """
		self.train_file_path = train_file_path
		self.test_file_path = test_file_path
		self.surprise_file_path = surprise_file_path
		self.predictions_file_path = predictions_file_path
		self.item_identifier = item_identifier
		self.svd = SVD()

	def make_surprise_file(self, user_min_occurrence=20, item_min_occurrence=20):
		"""Generates file to be ingested by Surprise.

	    Attributes:	        	        
	        user_min_occurrence: int
	            user must appear at least this number of times to be included

	        item_min_occurrence: int
	            item must appear at least this number of times to be included
		"""
		data = pd.read_csv(self.train_file_path)
		filtered_data = (
			data.groupby('user_id').filter(lambda x: len(x) >= user_min_occurrence)
				.groupby(self.item_identifier).filter(lambda x: len(x) >= item_min_occurrence)
			)
		print(filtered_data.shape)
		filtered_data.to_csv(path_or_buf=self.surprise_file_path, columns=['user_id', 'media_id', 'is_listened'], header=False, index=False)

	def make_file_if_missing(self):
		if not Path(self.surprise_file_path).is_file():
			print('File not found. Generating new input file')
			start_time = time.perf_counter()
			self.make_surprise_file()
			print('File generated in {}s'.format(time.perf_counter() - start_time))

	def read_data(self):
		reader = dataset.Reader(line_format="user item rating", sep=',', rating_scale=(0,1), skip_lines=1)
		self.data = dataset.Dataset.load_from_file(self.surprise_file_path, reader=reader)
		self.data.split(n_folds=5)

	def eval(self):
		# Evaluate performances of our algorithm on the dataset.
		perf = evaluate(self.svd, self.data, measures=['AUC'])
		print_perf(perf)

	def parameter_tuning(self):
		param_grid = {'n_epochs': [10, 20, 40], 'lr_all': [0.002, 0.005, 0.01],
		              'reg_all': [0.05, 0.1, 0.2]}

		print("Starting grid search...")
		start_time = time.perf_counter()
		self.grid_search = GridSearch(SVD, param_grid, measures=['AUC'])
		self.grid_search.evaluate(self.data)
		print('Grid search took {}s'.format(time.perf_counter() - start_time))

		self.svd = self.grid_search.best_estimator['AUC']

		print(self.grid_search.best_score['AUC'])
		print(self.grid_search.best_params['AUC'])

	def train(self):
		trainset = self.data.build_full_trainset()
		self.svd.train(trainset)

	def predict(self, user_lst, item_lst):
		assert len(user_lst) == len(item_lst)

		# gets predictions
		lst_length = len(user_lst)

		# true value is unknown but imputed as 0 for now
		pred = [self.svd.predict(str(user_lst[idx]), str(item_lst[idx]), 0) for idx in range(lst_length)]
		prediction, unseen = zip(*([(est, details['was_impossible']) for (_, _, _, est, details) in pred]))
		return prediction, unseen

	def write_test_predictions(self):
		data = pd.read_csv(self.test_file_path)
		user_lst, item_lst = data['user_id'].tolist(), data[self.item_identifier].tolist()
		predictions, unseen = self.predict(user_lst, item_lst)
		data = data.assign(pred=predictions, unseen=unseen)
		data.to_csv(path_or_buf=self.predictions_file_path, index=False)

if __name__ == '__main__':
	start_time = time.perf_counter()

	try:
		sfb = pickle.load(open( "sfb.p", "rb" ))
	except FileNotFoundError:
		sfb = SurpriseFeatureBuilder(TRAIN_FILE_PATH, TEST_FILE_PATH, SURPRISE_FILE_PATH, PREDICTIONS_FILE_PATH)
		sfb.make_file_if_missing()
		sfb.read_data()
		# sfb.parameter_tuning()
		sfb.eval()
		sfb.train()
		pickle.dump(sfb, open("sfb.p", "wb"))


	sfb.write_test_predictions()

