import pandas as pd
import numpy as np

import sys

from util.meta import full_split, cv1_split, cv1_split_time, test_split_time
from util import gen_prediction_name, gen_submission, score_sorted
from util.sklearn_model import SklearnModel
from util.keras_model import KerasModel
from util.xgb_model import XgbModel

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from scipy.special import logit


preds = [
    #'20170114-2122-ffm2-f1b-0.68827',
    #'20170114-0106-ffm2-f1b-0.68775',

    #'20170113-1506-ffm2-f1-0.68447',
    #'20170113-1213-ffm2-p1-0.68392',

    '20170110-0230-ffm2-f1-0.69220',
    '20170110-1055-ffm2-f1-2-0.69214',

    '20170110-0124-ffm2-f1-0.69175',

    '20170109-1354-ffm2-f1-0.69148',

    '20170108-2008-ffm2-f1-0.68984',

    '20170107-2248-ffm2-p1-0.68876',
    '20170108-0345-ffm2-p2-0.68762',

    '20170106-2000-ffm2-p1-0.68754',
    '20170106-2050-ffm2-p2-0.68656',

    '20170105-2113-ffm2-p1-0.68684',

    '20161230-1323-ffm-p1-0.68204',
    '20161230-1049-ffm-p2-0.68169',

    '20161231-0544-vw-p1-0.67309',
    '20161231-1927-vw-p2-0.66718',

    '20170106-1339-vw-p1-0.67829',
    '20170109-1239-vw-p2-0.67148',
]

models = {
    'lr': lambda: SklearnModel(LogisticRegression(C=0.01)),
    'nn': lambda: KerasModel(batch_size=128, layers=[40, 10], dropouts=[0.3, 0.1], n_epoch=1),
    'xgb': lambda: XgbModel(n_iter=1500, silent=1, objective='binary:logistic', eval_metric='logloss', seed=144, max_depth=4, colsample_bytree=0.5, subsample=0.25, tree_method='exact', eta=0.05)
}

model_name = sys.argv[1]
model_factory = models[model_name]


def y_hash(y):
    return hash(tuple(np.where(y[:200])[0]) + tuple(np.where(y[-200:])[0]))


def fit_present_model(events, train_X, train_y, train_event):
    print "Training present model..."

    train_is_present = train_event.isin(events[events['timestamp'] < cv1_split_time].index).values

    present_train_X = train_X[train_is_present].values
    present_train_y = train_y[train_is_present].values
    present_train_g = train_event[train_is_present].values

    folds = list(GroupKFold(3).split(present_train_X, present_train_y, present_train_g))
    ll_scores = []
    map_scores = []

    for k, (idx_train, idx_test) in enumerate(folds):
        fold_train_X = present_train_X[idx_train]
        fold_train_y = present_train_y[idx_train]
        fold_train_g = present_train_g[idx_train]

        fold_val_X = present_train_X[idx_test]
        fold_val_y = present_train_y[idx_test]
        fold_val_g = present_train_g[idx_test]

        model = model_factory()
        model.fit(fold_train_X, fold_train_y, fold_train_g, fold_val_X, fold_val_y, fold_val_g)

        pred = model.predict(fold_val_X)

        ll_scores.append(log_loss(fold_val_y, pred, eps=1e-7))
        map_scores.append(score_sorted(fold_val_y, pred, fold_val_g))

        print "    Fold %d logloss: %.7f, map score: %.7f" % (k+1, ll_scores[-1], map_scores[-1])

    print "  Present map score: %.7f +- %.7f" % (np.mean(map_scores), np.std(map_scores))

    return model_factory().fit(present_train_X, present_train_y, fold_train_g), np.mean(map_scores)


def fit_future_model(events, train_X, train_y, train_event):
    print "Training future model..."

    val2_split_time = 1078667779

    train_is_future_all = train_event.isin(events[events['timestamp'] >= cv1_split_time].index.values)
    train_is_future_train = train_event.isin(events[(events['timestamp'] >= cv1_split_time) & (events['timestamp'] < val2_split_time)].index.values)
    train_is_future_val = train_event.isin(events[(events['timestamp'] >= val2_split_time) & (events['timestamp'] < test_split_time)].index.values)

    future_train_X = train_X[train_is_future_train].values
    future_train_y = train_y[train_is_future_train].values
    future_train_g = train_event[train_is_future_train].values

    future_val_X = train_X[train_is_future_val].values
    future_val_y = train_y[train_is_future_val].values
    future_val_g = train_event[train_is_future_val].values

    model = model_factory()
    model.fit(future_train_X, future_train_y, future_train_g, future_val_X, future_val_y, future_val_g)

    pred = model.predict(future_val_X)

    ll_score = log_loss(future_val_y, pred, eps=1e-7)
    map_score = score_sorted(future_val_y, pred, future_val_g)

    print "  Future logloss: %.7f, map score: %.7f" % (ll_score, map_score)

    future_all_X = train_X[train_is_future_all].values
    future_all_y = train_y[train_is_future_all].values
    future_all_g = train_event[train_is_future_all].values

    return model_factory().fit(future_all_X, future_all_y, future_all_g), map_score


def load_x(ds):
    if ds == 'train':
        feature_ds = 'cv1_test'
        pred_ds = 'cv1'
    elif ds == 'test':
        feature_ds = 'full_test'
        pred_ds = 'test'
    else:
        raise ValueError()

    X = []
    X.append((pd.read_csv('cache/leak_%s.csv.gz' % feature_ds, dtype=np.uint8) > 0).astype(np.uint8))

    for pi, p in enumerate(preds):
        X.append(logit(pd.read_csv('preds/%s-%s.csv.gz' % (p, pred_ds), dtype=np.float32)[['pred']].rename(columns={'pred': 'p%d' % pi}).clip(lower=1e-7, upper=1-1e-7)))

    return pd.concat(X, axis=1)


def load_train_data():
    print "Loading train data..."

    d = pd.read_csv(cv1_split[1], dtype=np.uint32, usecols=['display_id', 'clicked'])

    return load_x('train'), d['clicked'], d['display_id']


## Main part


print "Loading events..."

events = pd.read_csv("../input/events.csv.gz", dtype=np.int32, index_col=0, usecols=[0, 3])  # Load events


## Training models

train_data = load_train_data()

present_model, present_score = fit_present_model(events, *train_data)
future_model, future_score = fit_future_model(events, *train_data)

score = present_score * 0.47671335657020786 + future_score * 0.5232866434297921

print "Estimated score: %.7f" % score

del train_data


## Predicting

print "Predicting on test..."
print "  Loading data..."

test_X = load_x('test').values

test_p = pd.read_csv(full_split[1], dtype=np.uint32)
test_p['pred'] = np.nan

test_is_present = test_p['display_id'].isin(events[events['timestamp'] < test_split_time].index).values
test_is_future = test_p['display_id'].isin(events[events['timestamp'] >= test_split_time].index).values

del events

print "  Predicting..."

name = gen_prediction_name('l2-%s' % model_name, score)

test_p.loc[test_is_present, 'pred'] = present_model.predict(test_X[test_is_present])
test_p.loc[test_is_future, 'pred'] = future_model.predict(test_X[test_is_future])

test_p[['pred']].to_csv('preds/%s-test.csv.gz' % name, index=False, compression='gzip')

del test_X, test_is_future, test_is_present

print "  Generating submission..."
subm = gen_submission(test_p)
subm.to_csv('subm/%s.csv.gz' % name, index=False, compression='gzip')

print "  File name: %s" % name
print "Done."
