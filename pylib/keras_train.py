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

model = Sequential()
model.add(BatchNormalization(mode=0, axis=1,input_shape=(120,4)))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(128, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(128, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(256, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(256, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(128, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(128, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(32, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(mode=0, axis=1))
model.add(Conv1D(32, 5, padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(30))

sgd = SGD(lr=0.000001, decay=0.0005, momentum=0.9, nesterov=True)
model.compile(loss=losses.mean_squared_error, optimizer=sgd)

checkpoint = ModelCheckpoint(filepath='mymodel.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

params = {'data_dir':'../data/day','batch_size':256, 'win_len':120, 'predict_len':30}
feeder = data.HSFeeder(params)
train_generator = feeder.generate_batch()
model.fit_generator(train_generator, steps_per_epoch = 100, epochs=1000, max_q_size=100, workers=5, callbacks=callbacks_list)