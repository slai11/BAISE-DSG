import sys
import os
import csv
import json
import time
import requests
import numpy as np
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioBasicIO

import torch

import pdb


"""
Purpose: Open or make dataset

Open -> open binary file and return train/test

Make -> download mp3, convert to wav, convert to tensor, delete mp3 and wav

"""

class Dataset(object):
    def __init__(self):
        self.preview_store = '../../data/deezer/track_api_preview.json'

    def open(self):
        self.store_dict = torch.load('store3.bin')
        self.tensor = self.store_dict.get('data')

        #generate random tensor for training
       
    def make(self):
        with open(self.preview_store) as data:
            preview_dict = json.load(data)
        
        track_ids = []
        all_tracks = []
        for i, track_id in enumerate(preview_dict):
            start = time.time()
            track_tensor = self._get_tensor(track_id, preview_dict.get(track_id))
            print((time.time()-start))
            if track_tensor is None:
                pass
            else:
                if track_tensor.shape == (768, 320):
                    all_tracks.append(track_tensor.T)
                    track_ids.append(int(track_id))
                    if len(track_ids) == 1000:
                        break
        
        all_tracks = np.array(all_tracks)
        all_tracks = torch.from_numpy(all_tracks).type(torch.FloatTensor)
        all_tracks = all_track.view(-1, 1, 320, 768)

        track_ids = np.array(track_ids)
        track_ids = torch.from_numpy(track_ids).type(torch.IntTensor)
        
        self.tensor = all_tracks

        store = {}
        store['ids'] = track_ids
        store['data'] = all_tracks
        torch.save(store, 'store.bin')
        pdb.set_trace()


    def _get_tensor(self, track_id, preview):
        if preview != 'NONE':
            track_name = '{}.mp3'.format(track_id)
            r = requests.get(preview)
            open(track_name, 'wb').write(r.content)

            wav_file = self._convert_to_wav(track_id)
            sp, _, _ = self._generate_spectrogram(wav_file)

            
            #delete both wav n mp3 to save storage
            os.remove(track_name)
            os.remove(wav_file)
            print("Removed MP3 and WAV file for {}".format(track_id))

            return sp
        else:
            return None

    
    def _generate_spectrogram(self, filename):
        
        [Fs, x] = audioBasicIO.readAudioFile(filename)
        x = audioBasicIO.stereo2mono(x)
        specgram, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs * 0.040), round(Fs * 0.040), False)
        return (specgram, TimeAxis, FreqAxis)

    def _convert_to_wav(self, track_id):
        Fs = 16000
        nC = 1
        mp3_file = '{}.mp3'.format(track_id)
        wav_file = '{}.wav'.format(track_id)
        command = "avconv -i \"" + mp3_file + "\" -ar " +str(Fs) + " -ac " + str(nC) + " \"" + wav_file + "\""
        os.system(command.decode('unicode_escape').encode('ascii','ignore').replace("\0",""))
        return wav_file

if __name__ == "__main__":
    ds = Dataset()
    ds.open()
    print(sys.getsizeof(ds.tensor))
