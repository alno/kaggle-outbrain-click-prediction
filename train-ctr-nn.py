import numpy as np
import pandas as pd

import argparse

from util.meta import row_counts
from util import train_model

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers


input_size = 19


def regularizer(params):
    if 'l1' in params and 'l2' in params:
        return regularizers.l1l2(params['l1'], params['l2'])
    elif 'l1' in params:
        return regularizers.l1(params['l1'])
    elif 'l2' in params:
        return regularizers.l2(params['l2'])
    else:
        return None


def nn_mlp_2(input_shape, **params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        model.add(PReLU())

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

    model.add(Dense(1, init='glorot_normal', activation='sigmoid'))

    return model


def read_data(name):
    return np.memmap("cache/%s_np_ctr1.npy" % name, dtype='float32', mode='r', shape=(row_counts[name], input_size))


def fit_predict(profile, split, split_name):
    train_X = read_data(split_name + '_train')
    train_y = pd.read_csv(split[0], usecols=['clicked'])['clicked'].values

    if split_name == 'full':
        eval_X = None
        eval_y = None
    else:
        eval_X = read_data(split_name + '_test')
        eval_y = pd.read_csv(split[1], usecols=['clicked'])['clicked'].values

    model = nn_mlp_2((input_size,), layers=[20])
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.fit(
        x=train_X, y=train_y,
        batch_size=256, nb_epoch=1,
        validation_data=(None if eval_X is None else (eval_X, eval_y)),
        verbose=1, callbacks=[])

    pred_X = read_data(split_name + '_test') if eval_X is None else eval_X
    pred = model.predict(pred_X, batch_size=256)

    pred_df = pd.read_csv(split[1])
    pred_df['pred'] = pred

    return pred_df


profile_name = 'v1'
profile = {}

train_model(fit_predict, 'ctr-nn-%s' % profile_name, profile)
