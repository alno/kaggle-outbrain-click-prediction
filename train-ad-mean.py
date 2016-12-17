import numpy as np
import pandas as pd

from util.meta import full_split, val_split
from util import gen_prediction_name, gen_submission, score_prediction

reg = 10.0


def fit_predict(train_file, pred_file):
    train = pd.read_csv(train_file, dtype=np.int32)
    train_pos = train[train['clicked'] == 1]

    ad_cnt = train['ad_id'].value_counts()
    ad_pos_cnt = train_pos['ad_id'].value_counts()

    del train, train_pos

    test = pd.read_csv(pred_file, dtype=np.int32)
    test['pred'] = test['ad_id'].map(ad_pos_cnt).fillna(0) / (test['ad_id'].map(ad_cnt).fillna(0) + reg)

    return test

## Validating

print "Running on validation split..."

pred = fit_predict(val_split[0], val_split[1])

print "Scoring..."

present_score, future_score, total_score = score_prediction(pred)
name = gen_prediction_name('ad-mean', total_score)

print "  Present score: %.5f" % present_score
print "  Future score: %.5f" % future_score
print "  Total score: %.5f" % total_score

pred[['display_id', 'ad_id', 'pred']].to_pickle('preds/%s-val.pickle' % name)

## Predicting

print "Running on full split..."

pred = fit_predict(full_split[0], full_split[1])
pred[['display_id', 'ad_id', 'pred']].to_pickle('preds/%s-test.pickle' % name)

print "  Generating submission..."
subm = gen_submission(pred)
subm.to_csv('subm/%s.csv.gz' % name, index=False, compression='gzip')

print "  File name: %s" % name
print "Done."
