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
import data
import numpy as np
import json
import sys, os

#model define
model = Sequential()
model.add(BatchNormalization(axis=1,input_shape=(200,4)))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(128, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(128, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(256, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(256, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(128, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(128, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(32, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(32, 5, padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))
#model define finished
json_string = model.to_json()
json.dump(json_string, open('mymodel.json','w'))

model_path = 'mymodel.h5'
if os.path.exists(model_path):
    model.load_weights(model_path)

sgd = SGD(lr=0.000001, decay=0.0005, momentum=0.9, nesterov=True)

model.compile(loss=losses.binary_crossentropy, optimizer=sgd)

checkpoint = ModelCheckpoint(filepath=model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

params = {'data_dir':'../data/day','batch_size':32, 'win_len':200, 'predict_len':100}
feeder = data.HSFeeder(params)
train_generator = feeder.generate_batch()
model.fit_generator(train_generator, steps_per_epoch = 1, epochs=100000, max_q_size=100, workers=5, callbacks=callbacks_list)