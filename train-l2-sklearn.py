import pandas as pd
import numpy as np

from util.meta import full_split, val_split, val_split_time, test_split_time
from util import gen_prediction_name, gen_submission, score_sorted

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

from scipy.special import logit


preds = [
    '20170106-2000-ffm2-p1-0.68754',
    '20170105-2113-ffm2-p1-0.68684',

    '20161230-1323-ffm-p1-0.68204',
    '20161230-1049-ffm-p2-0.68169',

    '20161231-0544-vw-p1-0.67309',
    '20161231-1927-vw-p2-0.66718',

    '20170106-1339-vw-p1-0.67829',
]


def create_model():
    return LogisticRegression(C=1.0)


def y_hash(y):
    return hash(tuple(np.where(y[:200])[0]) + tuple(np.where(y[-200:])[0]))


def fit_present_model(train_X, train_y, train_event):
    print "Training present model..."

    train_is_present = train_event.isin(events[events['timestamp'] < val_split_time].index).values

    present_train_X = train_X[train_is_present].values
    present_train_y = train_y[train_is_present].values
    present_train_g = train_event[train_is_present].values

    folds = list(GroupKFold(3).split(present_train_X, present_train_y, present_train_g))
    scores = []

    for k, (idx_train, idx_test) in enumerate(folds):
        model = create_model()
        model.fit(present_train_X[idx_train], present_train_y[idx_train])

        pred = model.predict_proba(present_train_X[idx_test])[:, 1]
        score = score_sorted(present_train_y[idx_test], pred, present_train_g[idx_test])

        print "    Fold %d score: %.7f" % (k+1, score)

        scores.append(score)

    print "  Present score: %.7f +- %.7f" % (np.mean(scores), np.std(scores))

    return create_model().fit(present_train_X, present_train_y), np.mean(scores)


def fit_future_model(train_X, train_y, train_event):
    print "Training future model..."

    val2_split_time = 1078667779

    train_is_future_all = train_event.isin(events[events['timestamp'] >= val_split_time].index.values)
    train_is_future_train = train_event.isin(events[(events['timestamp'] >= val_split_time) & (events['timestamp'] < val2_split_time)].index.values)
    train_is_future_val = train_event.isin(events[(events['timestamp'] >= val2_split_time) & (events['timestamp'] < test_split_time)].index.values)

    future_train_X = train_X[train_is_future_train].values
    future_train_y = train_y[train_is_future_train].values

    model = create_model()
    model.fit(future_train_X, future_train_y)

    future_val_X = train_X[train_is_future_val].values
    future_val_y = train_y[train_is_future_val].values
    future_val_g = train_event[train_is_future_val].values

    pred = model.predict_proba(future_val_X)[:, 1]
    score = score_sorted(future_val_y, pred, future_val_g)

    print "  Future score: %.7f" % score

    future_all_X = train_X[train_is_future_all].values
    future_all_y = train_y[train_is_future_all].values

    return create_model().fit(future_all_X, future_all_y), score


def load_x(ds):
    if ds == 'train':
        feature_ds = 'val_test'
        pred_ds = 'val'
    elif ds == 'test':
        feature_ds = 'full_test'
        pred_ds = 'test'
    else:
        raise ValueError()

    X = []
    X.append(pd.read_csv('cache/leak_%s.csv.gz' % feature_ds, dtype=np.uint8))

    for pi, p in enumerate(preds):
        X.append(logit(pd.read_pickle('preds/%s-%s.pickle' % (p, pred_ds))[['pred']].rename(columns={'pred': 'p%d' % pi})))

    return pd.concat(X, axis=1)


def load_train_data():
    print "Loading train data..."

    d = pd.read_csv(val_split[1], dtype=np.uint32, usecols=['display_id', 'clicked'])

    return load_x('train'), d['clicked'], d['display_id']


## Main part


print "Loading events..."

events = pd.read_csv("../input/events.csv.gz", dtype=np.int32, index_col=0, usecols=[0, 3])  # Load events


## Training models

train_data = load_train_data()

present_model, present_score = fit_present_model(*train_data)
future_model, future_score = fit_future_model(*train_data)

score = (present_score + future_score) / 2

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

name = gen_prediction_name('l2-lr', score)

test_p.loc[test_is_present, 'pred'] = present_model.predict_proba(test_X[test_is_present])[:, 1]
test_p.loc[test_is_future, 'pred'] = future_model.predict_proba(test_X[test_is_future])[:, 1]

test_p[['pred']].to_pickle('preds/%s-test.pickle' % name)

del test_X, test_is_future, test_is_present

print "  Generating submission..."
subm = gen_submission(test_p)
subm.to_csv('subm/%s.csv.gz' % name, index=False, compression='gzip')

print "  File name: %s" % name
print "Done."
