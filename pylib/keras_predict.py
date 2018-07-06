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

json_str = json.load(open('mymodel.json'))
model = model_from_json(json_str)

model_path = 'mymodel.h5'
if os.path.exists(model_path):
    model.load_weights(model_path)

params = {'data_dir':'../data/day','batch_size':256, 'win_len':120, 'predict_len':30}
feeder = data.HSFeeder(params)
train_generator = feeder.generate_batch()