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

params = {'data_dir':'../data/day','batch_size':32, 'win_len':200, 'predict_len':100}
feeder = data.HSFeeder(params)
batch_data= feeder.generate_one_sample()
X = batch_data['x']
Y = batch_data['y']
base = batch_data['base_price']
original_x = batch_data['original_x']
labels = batch_data['labels']

pred = model.predict(X)
pred = pred
Y = Y

print(pred)
print(labels)
for i in range(10):
    sample = original_x[i,:,1].reshape(-1,1)
    pp.figure(figsize = [300, 100])    
    pp.plot(range(200),sample, c='b')
    if pred[i] > 0.5:
        pp.plot(range(200,300),Y[i],c='r')
    else:
        pp.plot(range(200,300),Y[i],c='g')
    pp.show()