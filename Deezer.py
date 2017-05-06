import time
import requests
import json
import pdb
import csv
import os
import pandas as pd
import numpy as np
from queue import Queue
import threading

class DeezerAPI(object):
    """
    Pings Deezer API to get item information in a JSON object and stores it in 
    JSON file. Does it either in single or multhreaded way.

    Attributes:
        attr: string
            track, user, genre, album, artist
        col: string
            column name in train.csv
        api: string
            deezer api url
        attr_json: dict
            contains all attr objects

    """
    def __init__(self, attribute_name, colname):
        """
        Args:
            attribute_name: string
                used for api call
            colname: string
                name of column in dataframe
        """
        self.attr = attribute_name
        self.col = colname
        self.api = 'https://api.deezer.com'
        self.request_url = '/'.join([self.api, self.attr])
        
        self._load()

    # PUBLIC METHODS
    def get(self, item_id):
        """
        Args:
            item_id: string

        Return:
            dict: item's detail dictionary
        """
        return self.attr_json.get(item_id)
    
    def get_dict(self):
        return self.attr_json

    # INTERNAL METHODS
    def _load(self):
        """Loads attribute JSON if available, else download it using DeezerAPI

        check if file exist
        """
        filepath = 'data/deezer/{}_api.json'.format(self.attr)
        total = pd.read_csv('data/archive/train.csv')[self.col].unique().tolist()
        id_list = sorted(total, key=int)
 
        if os.path.exists(filepath):
            # open and add on
            with open(filepath) as data:
                self.attr_json = json.load(data)
            
            print("File exists with {} entries, will take from {}".format(len(self.attr_json), filepath))
            
            downloaded_id_list = list(self.attr_json.keys())            
            downloaded_id_list = list(filter(lambda x: x!='null', downloaded_id_list))
            downloaded_id_list = list(map(lambda x: int(x), downloaded_id_list))
            
            # Retain those that have yet to be downloaded
            id_list = list(set(total) - set(downloaded_id_list))
            assert((len(id_list) + len(downloaded_id_list) == len(total)))
        else:
            self.attr_json = {}
       
        if id_list: 
            #id_list = id_list[0:100000] # optional shortening of id-list
            self._download_w_multithread(id_list, filepath)
            #self._download_w_tornado(id_list, filepath)

    def _download(self, id_list, filepath):
        """
        Calls to deezer api and get info
        """
        try:
            start = time.time()
            for i, item_id in enumerate(id_list):
                query = '/'.join([self.request_url, str(item_id)])
                res = requests.get(query)
                try:
                    res = res.json()
                except:
                    pdb.set_trace()
                self.attr_json[res.get('id')] = res

                print(i)
            
                if i % 1000 == 0 and i > 0:
                    print("Took {}s to run 1000 queries -- wrote and saved {} more to json".format((time.time() - start), i))
                    start = time.time()
                    with open(filepath, 'w') as outfile:
                        json.dump(self.attr_json, outfile)
        except KeyboardInterrupt:
            with open(filepath, 'w') as outfile:
                json.dump(self.attr_json, outfile)
    
    def _download_w_multithread(self, id_list, filepath):
        """Calls to deezer api and get info using 10 threads

        Args:
            id_list: list of int
            filepath: JSON filepath
        """
        attr_json = {}
        class MyThread(threading.Thread):
            """threading object"""
            def __init__(self, inputqueue):
                threading.Thread.__init__(self)
                self.queue = inputqueue
            
            def run(self):
                """"Runs download one time from queue"""
                while True:
                    item, request = self.queue.get()
                    self.__download_single(item, request)
                    self.queue.task_done()

            def __download_single(self, item_id, request):
                """Downloads 1 item's worth of detail and load into dict """
                MAX_RETRIES = 5
                #session = requests.Session()
                #adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
                #session.mount('https://', adapter)
                #session.mount('http://', adapter)
                try:
                    query = '/'.join([request, str(item_id)])
                    res = requests.get(query)
                    res = res.json()
                    attr_json[res.get('id')] = res
                except:
                    print("{} needs to redo".format(item_id))
                    time.sleep(1)
                    self.__download_single(item_id, request)

        queue = Queue()
        
        # Initialise threads
        for i in range(10):
            thread = MyThread(queue)
            thread.daemon = True
            thread.start()
        print("Created 10 threads")

        # Push all into queue
        for i, item_id in enumerate(id_list):
            queue.put((item_id,self.request_url))
                        
        print("Assigned jobs to queue")
        try:
        # Checks length and input to JSON file every 1 minute
            while queue.qsize() != 0:
                time.sleep(10)
                length = float(len(attr_json))
                
                print(length)
                if length > 1000 and length % 10000 < 100:
                    print('Saving JSON cuz i kiasu')
                    self.attr_json.update(attr_json)
                    with open(filepath, 'w') as outfile:
                        json.dump(self.attr_json, outfile)
 
            
            print("DONE")
            self.attr_json.update(attr_json)
            print("Writing file of length {}".format(len(self.attr_json)))
            with open(filepath, 'w') as outfile:
                json.dump(self.attr_json, outfile)
            print("File written")
        
        except KeyboardInterrupt:
            self.attr_json.update(attr_json)
            print("Writing file of length {}".format(len(self.attr_json)))
            with open(filepath, 'w') as outfile:
                json.dump(self.attr_json, outfile)
            print("Safe to exit")
        
if __name__ == '__main__':
   # deezer = DeezerAPI('track', 'media_id')
   # deezer = DeezerAPI('user', 'user_id') #1 guy do this   
    #deezer = DeezerAPI('album', 'album_id') #1 guy do this
    #deezer = DeezerAPI('artist', 'artist_id') #1 guy do this
    #deezer = DeezerAPI('genre', 'genre_id') #1 guy do this



