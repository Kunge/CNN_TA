import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
import data
import numpy as np
import json
import sys, os

#model define
K.set_learning_phase(1)
model = Sequential()
model.add(BatchNormalization(axis=1,input_shape=(120,4)))
model.add(MaxPooling1D(pool_size=5, strides = 2))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=1))
model.add(Conv1D(64, 5, padding='valid'))
model.add(Activation('relu'))

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
model.add(Flatten())
model.add(Activation('relu'))

model.add(BatchNormalization())
model.add(Dense(50))

model.add(BatchNormalization())
model.add(Dense(3,activation='softmax'))
#model define finished
json_string = model.to_json()
json.dump(json_string, open('mymodel.json','w'))

model_path = 'mymodel.h5'
if os.path.exists(model_path):
    model.load_weights(model_path)

sgd = SGD(lr=0.00001, decay=0.0005, momentum=0.9, nesterov=True)

model.compile(loss=losses.categorical_crossentropy, optimizer=sgd)

checkpoint = ModelCheckpoint(filepath=model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

params = {'data_dir':'../data/','batch_size':128, 'win_len':120, 'predict_len':200}
feeder = data.HSFeeder(params)
train_generator = feeder.generate_batch()
model.fit_generator(train_generator, steps_per_epoch = 1, epochs=100000, workers=1, callbacks=callbacks_list)