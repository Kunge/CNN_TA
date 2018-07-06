import tensorflow as tf
import keras.backend.tensorflow_backend as K

from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import data
import numpy as np
import json
import sys, os
import matplotlib.pyplot as pp

json_str = json.load(open('mymodel.json'))
model = model_from_json(json_str)

model_path = 'mymodel.h5'
if os.path.exists(model_path):
    model.load_weights(model_path)

params = {'data_dir':'../data/day','batch_size':32, 'win_len':100, 'predict_len':200}
feeder = data.HSFeeder(params)
batch_data= feeder.generate_one_sample()
X = batch_data['x']
Y = batch_data['y']
base = batch_data['base_price']
original_x = batch_data['original_x']

preds = model.predict(X).reshape(-1)
Y = Y
print(preds)
i = -1
high = 0.8
low = 0.3
for pred in preds:
    i = i + 1
    sample = original_x[i,:,1].reshape(-1,1)
    pp.figure(figsize = [300, 100])    
    pp.plot(range(100),sample, c='b')
    if pred > high:
        pp.plot(range(100,300),Y[i],c='r')
    if pred < low:
        pp.plot(range(100,300),Y[i],c='g')
    if pred > low and pred < high:
        pp.plot(range(100,300),Y[i],c='y')
    pp.show()