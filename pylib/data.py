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
        sample_size = int(len(self._filename_list)*0.8)
        self._training_files = np.random.choice( self._filename_list, size = sample_size )
        #self._dataset = self._load_data()
    
    def _get_filename_list(self, data_dir):
        filename_list = list()
        for root, dirs, files in os.walk( self._data_dir ):
            for name in files:
                filename_list.append( os.path.join(root, name) )
        return filename_list

    def _load_data(self, sample_files):
        dataset = dict()
        new_sample_files = []
        for filename in sample_files:
            data = json.load( open(filename) )
            if len(data)>self._win_len+self._predict_len+10:
                dataset[filename] = data
                new_sample_files.append(filename)
        return dataset, new_sample_files
    
    def generate_one_sample(self):
        sample_file = np.random.choice( self._training_files, size = 1 )
        dataset, sample_file = self._load_data(sample_file)
        batch_data = self._get_batch_data(sample_file,dataset)
        return batch_data['x'], batch_data['y'], batch_data['base_price']

    def generate_batch(self):
        while True:
            sample_files = np.random.choice( self._training_files, size = self._batch_size*2 )
            dataset, sample_files = self._load_data(sample_files)
            batch_data = self._get_batch_data(sample_files,dataset)
            yield batch_data['x'], batch_data['labels']

    def _get_batch_data(self, sample_files, dataset):
        batch_data = dict()
        training = []
        target = []
        labels = []
        i = 0
        while i < self._batch_size:
            filename = np.random.choice( sample_files )
            data = dataset[filename]
            length = len(data)
            start = np.random.choice( length-self._win_len-self._predict_len-1 )
            training_data = data[start:start+self._win_len]
            target_data = data[start+self._win_len:start+self._win_len+self._predict_len]
            base_price = training_data[-1]['close']
            y=[]
            for k in target_data:
                c = k['close']
                y.append(c)
            label = -1
            if max(y)>base_price*1.8 and min(y)>base_price*0.8:
                label = 1
            if min(y)<base_price*0.6 and max(y)<base_price*1.2:
                label = 0
            
            if label < 0:
                continue
            labels.append(label)
            i = i +1
            x=[]
            for k in training_data:
                o = k['open']
                c = k['close']
                h = k['high']
                l = k['low']
                a = np.array([o,c,h,l]).reshape([-1,1])
                x.append(a)
            x = np.squeeze(x)
            x[:,0] = x[:,0]-np.mean(x[:,0])
            x[:,1] = x[:,1]-np.mean(x[:,1])
            x[:,2] = x[:,2]-np.mean(x[:,2])
            x[:,3] = x[:,3]-np.mean(x[:,3])
        
            y = np.array(y)
            y = y-base_price
            training.append(x)
            target.append(y)
        batch_data['x'] = np.array(training)
        batch_data['y'] = np.array(target)
        batch_data['labels'] = np.array(labels)
        batch_data['base_price'] = training_data[-1]['close']
        return batch_data

if __name__ == '__main__':
    params = {'data_dir':'../data/day','batch_size':10, 'win_len':100, 'predict_len':30}
    feeder = HSFeeder(params)
    batch_data = feeder._get_batch_data()
    print(batch_data['x'])
    print(batch_data['y'])
    print(batch_data['x'].shape)