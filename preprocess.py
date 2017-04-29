import csv
import json
import numpy as np
from scipy.sparse import lil_matrix

import pandas as pd
import pdb
import time

from sys import getsizeof

class MatrixMaker(object):
    """MatrixMaker takes in a N-by-3 array and pivots it
    Attributes:
        threshold: int
            minimum observations of a variable value
        
        col: list of string
            column names from pandas df
        
        array: N by 3 numpy array
            var1, var2, is_listened
        
        matrix: 2d numpy array
            1s, 0s and np.nans

        post_matrix: 2d numpy array
            'answer' from whatever MF process

        row_map: dict
            key: id, value: index
        
        col_map: dict
            same as row_map
    """
    def __init__(self, array, columns, threshold=50, mode='scipy'):
        """
        columns, first one is Y axis, 2nd one is X axis
        Args:
            array: pandas dataframe. fullsized
            columns: list of 2 string
                [row_variable, column_variable]
            threshold: int, default 50
                determines the minimum number of times a variable 
                value has to appear


        """
        self.threshold = threshold
        columns.append('is_listened')
        self.col = columns
        self.array = array[self.col].as_matrix()
        self.post_matrix = None
        self.mode = mode

        self._build()



    def get_matrix(self):
        return self.matrix

    def store_post_fac_matrix(self, matrix):
        self.post_matrix = matrix

    def get_value(self, row_idx, col_idx):
        i = self.row_map.get(row_idx)
        j = self.col_map.get(col_idx)

        if self.post_matrix:
            return self.post_matrix[i,j]
        else:
            print('Post-factorised matrix not stored')
            return None

    def _build(self):
        """filter by threshold level then builds output matrix
        """
        # keep ids that pass the threshold
        row = list(map(lambda x: x[0], self.array))
        col = list(map(lambda x: x[1], self.array))        
        row = self._apply_threshold(row)
        col = self._apply_threshold(col)
       
        # creates matrix
        self._set_dict(row, col)
        self._set_output_matrix()
    
    def _apply_threshold(self, array):
        """Performs binning to filter by frequency of row_item and col_item

        Args:
            array: 1d list
        
        Return:
            filtered_list: 1d list of int
        """
        array = np.array(array).astype(int)
        array = array.reshape((array.shape[0],))
        count = np.bincount(array)
        ii = np.nonzero(count)[0]
        
        freq_table = list(zip(ii, count[ii])) 
        filtered_table = list(filter(lambda x: x[1] > self.threshold, freq_table))
        filtered_list = list(map(lambda x: x[0], filtered_table))
        return filtered_list

    def _set_output_matrix(self):
        """Input n x 3 array into row x col array
        """
        shape = (len(self.row_map), len(self.col_map))
        
        if self.mode == 'scipy':
            output = np.zeros(shape, dtype=float)
            output.fill(np.nan) # fill with 0.01??
            extra = 0
        else:
            output = lil_matrix(shape)
            extra = 1

        for obs in self.array:
            i = self.row_map.get(obs[0])
            j = self.col_map.get(obs[1])
            if i and j:   
                output[i,j] = int(obs[2]) + extra
        
        self.matrix = output

    def _set_dict(self, row, col):
        """
        Sets the dictionary to map id to index in array
        sets 2 dictionary
        """
        self.row_map = {}
        self.col_map = {}
        
        for i, x in enumerate(row):
            self.row_map[x] = i
        for j, y in enumerate(col):
            self.col_map[y] = j



if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    mm = MatrixMaker(df, ['user_id', 'media_id'], mode='numpy')
    np_output = mm.get_matrix()
    print(np_output)
    print(getsizeof(np_output))

    scmm = MatrixMaker(df, ['user_id', 'media_id'], mode='scipy')
    sc_output = scmm.get_matrix()
    print(getsizeof(sc_output))
    pdb.set_trace()




