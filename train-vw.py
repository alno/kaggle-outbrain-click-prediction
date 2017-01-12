import numpy as np
import pandas as pd

import os
import argparse

from scipy.special import expit

from util.meta import full_split, val_split
from util import gen_prediction_name, gen_submission, score_prediction, print_and_exec


def fit_predict(profile, split, split_name):
    train_file = 'cache/%s_train_vw.txt' % split_name
    pred_file = 'cache/%s_test_vw.txt' % split_name

    print "  Training..."

    if os.path.exists(train_file + '.cache'):
        os.remove(train_file + '.cache')

    interactions = ' '.join('-q %s' % i for i in profile['interactions'].split(' '))

    print_and_exec("vw --cache -P 5000000 --loss_function logistic %s %s -f /tmp/vw.model %s " % (profile['options'], interactions, train_file))

    print "  Predicting..."

    if os.path.exists(pred_file + '.cache'):
        os.remove(pred_file + '.cache')

    print_and_exec("vw -i /tmp/vw.model -p /tmp/vw.preds -P 5000000 -t %s" % pred_file)

    pred = pd.read_csv(split[1])
    pred['pred'] = expit(np.loadtxt('/tmp/vw.preds'))

    return pred


profiles = {
    'p1': {
        'interactions': 'aa al ld lp dp fe fa fd fl fp ff',
        'options': "--passes 3 -b 22 --nn 10 --ignore u --ignore t",
    },

    'p2': {
        'interactions': 'aa al ld lp dp ft fa fd fl fp ff tt ta tl td tp up',
        'options': "--passes 4 -b 23 --nn 20",
    }
}


parser = argparse.ArgumentParser(description='Train VW model')
parser.add_argument('profile', type=str, help='Train profile')
parser.add_argument('--rewrite-cache', action='store_true', help='Drop cache files prior to train')

args = parser.parse_args()
profile = profiles[args.profile]


if not os.path.exists('cache/val_train_vw.txt') or args.rewrite_cache:
    print "Generating data..."
    os.system("bin/export-vw-data")


## Validation

print "Validation split..."

pred = fit_predict(profile, val_split, 'val')

print "  Scoring..."

present_score, future_score, score = score_prediction(pred)
name = gen_prediction_name('vw-%s' % args.profile, score)

print "  Present score: %.5f" % present_score
print "  Future score: %.5f" % future_score
print "  Total score: %.5f" % score

pred[['pred']].to_csv('preds/%s-val.csv.gz' % name, index=False, compression='gzip')

del pred

## Prediction

print "Full split..."

pred = fit_predict(profile, full_split, 'full')
pred[['pred']].to_csv('preds/%s-test.csv.gz' % name, index=False, compression='gzip')

print "  Generating submission..."
subm = gen_submission(pred)
subm.to_csv('subm/%s.csv.gz' % name, index=False, compression='gzip')

del pred, subm

print "  File name: %s" % name
print "Done."
