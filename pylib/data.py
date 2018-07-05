#load the data feeding to model
import numpy as np
import json
import os,sys
import time


class HSFeeder:
    def __init__(self,params):
        self._data_dir = params['data_dir']
        self._batch_size = params['batch_size']
        self._win_len = params['win_len']#training window size
        self._predict_len = params['predict_len']#prediction window size

        self._filename_list = self._get_filename_list(self._data_dir)
        sample_size = 10#int(len(self._filename_list)*0.6)
        self._training_files = np.random.choice( self._filename_list, size = sample_size )
        #self._dataset = self._load_data()
    
    def _get_filename_list(self, data_dir):
        filename_list = list()
        for root, dirs, files in os.walk( self._data_dir ):
            for name in files:
                filename_list.append( os.path.join(root, name) )
        return filename_list

    def _load_data(self):
        dataset = dict()
        for filename in self._filename_list:
            data = json.load( open(filename) )
            dataset[filename] = data
            time.sleep(0.01)
        return dataset

    def generate_batch(self):
        while True:
            batch_data = self._get_batch_data()
            yield batch_data['x'], batch_data['y']

    def _get_batch_data(self):
        batch_data = dict()
        training = []
        target = []
        for i in range(self._batch_size):
            length = 0
            while length < self._win_len+self._predict_len+10:
                filename = np.random.choice( self._training_files )
                data = json.load( open(filename) )
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
                x.append(a)
            x = np.squeeze(x)
            y=[]
            for k in target_data:
                c = k['close']
                y.append(c)
            training.append(x)
            target.append(y)
        batch_data['x'] = np.array(training)
        batch_data['y'] = np.array(target)
        return batch_data

if __name__ == '__main__':
    params = {'data_dir':'../data/day','batch_size':10, 'win_len':100, 'predict_len':30}
    feeder = HSFeeder(params)
    batch_data = feeder._get_batch_data()
    print(batch_data['x'])
    print(batch_data['y'])
    print(batch_data['x'].shape)