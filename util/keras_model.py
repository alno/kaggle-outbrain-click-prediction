from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers

from sklearn.preprocessing import StandardScaler


def regularizer(params):
    if 'l1' in params and 'l2' in params:
        return regularizers.l1l2(params['l1'], params['l2'])
    elif 'l1' in params:
        return regularizers.l1(params['l1'])
    elif 'l2' in params:
        return regularizers.l2(params['l2'])
    else:
        return None


def nn_mlp_2(input_shape, params):
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


class KerasModel(object):

    def __init__(self, **params):
        self.arch = nn_mlp_2
        self.params = params

    def fit(self, train_X, train_y, train_g=None, eval_X=None, eval_y=None, eval_g=None):
        self.model = self.arch((train_X.shape[1],), self.params)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.scaler = StandardScaler()

        train_X = self.scaler.fit_transform(train_X)

        if eval_X is not None:
            eval_X = self.scaler.transform(eval_X)

        callbacks = []

        self.model.fit(
            x=train_X, y=train_y,
            batch_size=self.params.get('batch_size', 32), nb_epoch=self.params['n_epoch'],
            validation_data=(None if eval_X is None else (eval_X, eval_y)),
            verbose=1, callbacks=callbacks)

        return self

    def predict(self, test_X):
        test_X = self.scaler.transform(test_X)

        return self.model.predict(test_X).flatten()
