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

params = {'data_dir':'../data/day','batch_size':32, 'win_len':120, 'predict_len':30}
feeder = data.HSFeeder(params)
X,Y, base = feeder.generate_one_sample()
pred = model.predict(X)
pred = pred+base
Y = Y+base

for i in range(10):
    pp.plot(pred[i],c='r')
    pp.plot(Y[i],c='g')
    pp.show()