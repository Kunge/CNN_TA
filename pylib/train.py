import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
import functools

import logging
import data
import json
import model

slim = tf.contrib.slim


model = model.CNN_TA()

class MonitorHook(tf.train.SessionRunHook):
    def __init__(self):
        pass
    def begin(self):
        pass
    def before_run(self):
        fetches = {}
        graph = tf.get_default_graph()
        if self._global_step is None:
            fetches['global_step'] = graph.get_tensor_by_name('global_step:0')
        if self._global_step is None or\
                self._global_step % self._every_n_iter == 0:
            for name in self._periodic_keys:
                fetches[name] = graph.get_tensor_by_name(name + ':0')
        for name in self._regular_keys:
            fetches[name] = graph.get_tensor_by_name(name + ':0')
        return tf.train.SessionRunArgs(fetches)

    def after_run(self):
        fetches = run_values.results
        if fetches.has_key('global_step'):
            self._global_step = fetches['global_step']
        else:
            assert self._global_step is not None

        if self._global_step % self._every_n_iter == 0:
            np.save(self._cache_dir + '/' + self._dump_name + '.npy', fetches)
        self._global_step += 1


def get_train_op(features, labels, total_loss, params):
    batch_size = params.batch_size
    loss = total_loss

    global_step = tf.train.get_global_step()
    tf.identity(global_step, 'global_step')
    starter_learning_rate = 1e-3
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
            global_step, 100000, 0.5, staircase=True)
    tf.identity(learning_rate, 'learning_rate')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = slim.learning.create_train_op(
                    loss,
                    optimizer,
                    clip_gradient_norm=1.0,
                    global_step=global_step)
    return train_op

def get_input(generator, params):
    output_types = {
        'key1':tf.float32,
        'key2':tf.int32
    }
    dataset = tf.data.Dataset.from_generator(generator,\
            output_types = output_types)
    dataset = dataset.map( lambda x : x, num_parallel_calls = 5 )
    dataset = dataset.prefetch(buffer_size = 100)
    next_elements = dataset.make_one_shot_iterator().get_next()
    feature_keys = ['key1']
    label_keys = ['key2']
    features = { key: next_elements[key] for key in feature_keys }
    labels = { key: next_elements[key] for key in label_keys }
    return features, labels

def eval_metric_operate(loss_dict):
    pass

def model_fn(features, labels):
    tf.keras.backend.set_learning_phase(True)
    predictions = model.build(features)
    loss = None
    train_op = None
    eval_metric_ops = None

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
        loss_dict, total_loss = model.loss(predictions, labels)
        loss = toal_loss
        eval_metric_ops = eval_metric_operate(loss_dict)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = get_train_op(features, labels, total_loss, params)

    export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
    }

    spec = tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            loss = loss,
            train_op = train_op,
            eval_metric_ops = eval_metric_ops,
            export_outputs = export_outputs)
    return spec


def get_estimator(run_config, params):
     return tf.estimator.Estimator(
            model_fn = model_fn,
            params = params,
            config = run_config)


def train():
    params = tf.contrib.training.HParams(
        monitor_every_n = 100,
        evaluate_every_n = 1000
    )
    params.add_hparam('model_dir', os.path.abspath(tf.flags.FLAGS.model_dir))
    if not os.path.exists(tf.flags.FLAGS.model_dir):
        os.makedirs(tf.flags.FLAGS.model_dir)
    
    with open(tf.flags.FLAGS.model_dir + '/params.json', 'w') as f:
        f.write(params.to_json(indent = 4))
    feeder = data.HSFeeder(params)
    
    run_config = tf.contrib.learn.RunConfig( model_dir = params.model_dir )
    estimator = get_estimator(run_config, params)
    train_input_fn = functools.partial(get_input_fn, feeder.generate_batch, params)

    while True:
        if estimator.latest_checkpoint() is not None:
            global_step = estimator.get_variable_value('global_step')
        else:
            global_step = 0
	    tensors_to_log = {
                'total_loss':'total_loss'
                }
        logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,
                every_n_iter = params.monitor_every_n)
        train_monitor = MonitorHook('train_fetches', params, feeder = feeder)
        
        estimator.train(input_fn = train_input_fn,
		        hooks = [logging_hook, train_monitor],
                steps = params.evaluate_every_n -\
                        (global_step % params.evaluate_every_n))

if __name__ == '__main__':
    train()