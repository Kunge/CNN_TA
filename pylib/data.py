#load the data feeding to model
import numpy as np
import json
import os,sys


class HSFeeder:
    def __init__(self,params):
        self._data_dir = params.data_dir
        self._batch_size = params.batch_size
        self._win_len = params.win_len#training window size
        self._predict_len = params.predict_len#prediction window size

        self._filename_list = self._get_filename_list(self._data_dir)
        sampel_size = int(len(self._filename_list)*0.6)
        self._training_files = np.random.choice( self._filename_list, size = sample_size )
        self._dataset = self._load_data()
    
    def _get_filename_list(self, data_dir):
        filename_list = list()
        for root, dirs, files in os.walk( self._data_dir ):
            for name in files:
                filename_list.append( os.path.join(root, name) )
        return filename_list

    def _load_data(self, filename_list = self._filename_list):
        dataset = dict()
        for filename in filename_list:
            data = json.load( open(filename) )
            dataset[filename] = data

    def generate_batch(self):
        while True:
            batch_data = self._get_batch_data()
            yield batch_data

    def _get_batch_data(self):
        batch_data = dict()
        training = []
        target = []
        for i in range(self._batch_size):
            filename = np.random.choice( self._training_files )
            data = self._dataset[filename]
            length = len(data)
            start = np.random.choice( length-self._win_len-self._predict_len-1 )
            training_data = data[start:start+self._win_len]
            target_data = data[start+self._win_len:start+self._win_len+self._predict_len]
            x=[]
            for k in training_data:
                o = k['open']
                c = k['close']
                h = k['high']
                l = k['low']
                a = np.array([o,c,h,l]).reshape([-1,1])
                x.append( a )
            y=[]
            for k in target_data:
                c = k['close']
                y.append(c)
            training.append(x)
            target.append(y)
        batch_data['x'] = training
        batch_data['y'] = target
        return batch_data