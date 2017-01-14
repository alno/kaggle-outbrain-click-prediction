import numpy as np
import pandas as pd

import os
import argparse

from util.meta import full_split, cv1_split, cv2_split
from util import gen_prediction_name, gen_submission, score_prediction, print_and_exec


def fit_predict(profile, split, split_name):
    train_file = 'cache/%s_train_bin_%s' % (split_name, profile['dataset'])
    pred_file = 'cache/%s_test_bin_%s' % (split_name, profile['dataset'])

    n_bags = profile.get('bags', 1)

    pred = None
    for i in xrange(n_bags):
        opts = profile.get('options', '')
        opts += " --seed %d --epochs %d" % (profile['seed'] + i * 3407, profile['epochs'])

        if split_name != "full":
            opts += " --val %s" % pred_file

        print_and_exec("bin/ffm %s --train %s --test %s --pred  /tmp/ffm2.preds" % (opts, train_file, pred_file))

        if pred is None:
            pred = np.loadtxt('/tmp/ffm2.preds')
        else:
            pred += np.loadtxt('/tmp/ffm2.preds')

    pred_df = pd.read_csv(split[1])
    pred_df['pred'] = pred / n_bags

    return pred_df


profiles = {
    'p1': {
        'epochs': 4,
        'seed': 2017,
        'dataset': "p1",
    },

    'p1r': {
        'epochs': 4,
        'seed': 123,
        'options': "--restricted",
        'dataset': "p1",
    },

    'f1': {
        'epochs': 4,
        'seed': 42,
        'dataset': "f1",
    },

    'f1b': {
        'bags': 3,
        'epochs': 4,
        'seed': 178,
        'dataset': "f1",
    },

    'f1r': {
        'epochs': 4,
        'seed': 71,
        'options': "--restricted",
        'dataset': "f1",
    },

    'f2': {
        'epochs': 4,
        'seed': 456,
        'dataset': "f2",
    },

    'f2r': {
        'epochs': 4,
        'seed': 879,
        'options': "--restricted",
        'dataset': "f2",
    },
}


parser = argparse.ArgumentParser(description='Train FFM2 model')
parser.add_argument('profile', type=str, help='Train profile')
parser.add_argument('--rewrite-cache', action='store_true', help='Drop cache files prior to train')

args = parser.parse_args()
profile = profiles[args.profile]


if not os.path.exists('cache/full_train_bin_%s.index' % profile['dataset']) or args.rewrite_cache:
    print "Generating data..."
    os.system("bin/export-bin-data-%s" % profile['dataset'])


## Validation on CV2

print "CV2 split..."

pred = fit_predict(profile, cv2_split, 'cv2')

print "  Scoring..."

cv2_present_score, cv2_future_score, cv2_score = score_prediction(pred)
name = gen_prediction_name('ffm2-%s' % args.profile, cv2_score)

print "  Present score: %.5f" % cv2_present_score
print "  Future score: %.5f" % cv2_future_score
print "  Total score: %.5f" % cv2_score

pred[['pred']].to_csv('preds/%s-cv2.csv.gz' % name, index=False, compression='gzip')

del pred

## Validation on CV1

print "CV1 split..."

pred = fit_predict(profile, cv1_split, 'cv1')

print "  Scoring..."

cv1_present_score, cv1_future_score, cv1_score = score_prediction(pred)

print "  Present score: %.5f" % cv1_present_score
print "  Future score: %.5f" % cv1_future_score
print "  Total score: %.5f" % cv1_score

pred[['pred']].to_csv('preds/%s-cv1.csv.gz' % name, index=False, compression='gzip')

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
