import numpy as np
import pandas as pd

import os
import argparse

from scipy.special import expit

from util.meta import full_split, val_split
from util import gen_prediction_name, gen_submission, score_prediction, print_and_exec


def fit_predict(split, split_name):
    train_file = 'cache/%s_train_ffm.txt' % split_name
    pred_file = 'cache/%s_test_ffm.txt' % split_name

    print "  Training..."

    if os.path.exists(train_file + '.cache'):
        os.remove(train_file + '.cache')

    opts = "-t 3 -s 4"

    if split_name == "val":
        opts += " -p cache/val_test_ffm.txt"

    os.system("ffm-train %s %s ffm.model" % (opts, train_file))

    print "  Predicting..."

    if os.path.exists(pred_file + '.cache'):
        os.remove(pred_file + '.cache')

    print_and_exec("ffm-predict %s ffm.model ffm.preds" % pred_file)

    pred = pd.read_csv(split[1])
    pred['pred'] = expit(np.loadtxt('ffm.preds'))

    return pred


parser = argparse.ArgumentParser(description='Train FFM model')
parser.add_argument('--rewrite-cache', action='store_true', help='Drop cache files prior to train')

args = parser.parse_args()


if not os.path.exists('cache/val_train_ffm.txt') or args.rewrite_cache:
    print "Generating data..."
    os.system("bin/export-ffm-data")


## Validation

print "Validation split..."

pred = fit_predict(val_split, 'val')

print "  Scoring..."

present_score, future_score, score = score_prediction(pred)
name = gen_prediction_name('ffm', score)

print "  Present score: %.5f" % present_score
print "  Future score: %.5f" % future_score
print "  Total score: %.5f" % score

pred[['display_id', 'ad_id', 'pred']].to_pickle('preds/%s-val.pickle' % name)

del pred

## Prediction

print "Full split..."

pred = fit_predict(full_split, 'full')
pred[['display_id', 'ad_id', 'pred']].to_pickle('preds/%s-test.pickle' % name)

print "  Generating submission..."
subm = gen_submission(pred)
subm.to_csv('subm/%s.csv.gz' % name, index=False, compression='gzip')

del pred, subm

print "  File name: %s" % name
print "Done."
