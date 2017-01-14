import numpy as np
import xgboost as xgb

from sklearn.metrics import log_loss

from . import score_sorted


def y_hash(y):
    return hash(tuple(np.where(y[:200])[0]) + tuple(np.where(y[-200:])[0]))


class XgbModel(object):

    def __init__(self, n_iter, **params):
        self.n_iter = n_iter
        self.params = params

    def fit(self, train_X, train_y, train_g=None, eval_X=None, eval_y=None, eval_g=None):
        params = self.params.copy()

        dtrain = xgb.DMatrix(train_X, label=train_y)

        groups = {y_hash(train_y): train_g}

        if eval_X is None:
            watchlist = [(dtrain, 'train')]
        else:
            deval = xgb.DMatrix(eval_X, label=eval_y)
            watchlist = [(deval, 'eval'), (dtrain, 'train')]
            groups[y_hash(eval_y)] = eval_g

        def feval(y_pred, dtrain):
            y_true = dtrain.get_label()
            y_group = groups[y_hash(y_true)]

            return [
                ('loss', log_loss(y_true, y_pred)),
                ('map', score_sorted(y_true, y_pred, y_group)),
            ]

        self.model = xgb.train(params, dtrain, self.n_iter, watchlist, feval=feval, verbose_eval=20)

        return self

    def predict(self, test_X):
        return self.model.predict(xgb.DMatrix(test_X))
