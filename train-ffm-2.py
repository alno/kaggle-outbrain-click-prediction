import numpy as np
import pandas as pd

import os
import argparse

from util import print_and_exec, train_model


def fit_predict(profile, split, split_name):
    train_file = 'cache/%s_train_bin_%s' % (split_name, profile['dataset'])
    pred_file = 'cache/%s_test_bin_%s' % (split_name, profile['dataset'])

    n_bags = profile.get('bags', 1)

    pred = None
    for i in xrange(n_bags):
        opts = profile.get('options', '')
        opts += " --seed %d --epochs %d" % (profile.get('seed', np.random.randint(1e6)) + i * 3407, profile['epochs'])

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

    'p1b': {
        'epochs': 4,
        'bags': 3,
        'dataset': "p1",
    },

    'f1': {
        'epochs': 4,
        'seed': 42,
        'dataset': "f1",
    },

    'f1b': {
        'bags': 2,
        'epochs': 5,
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

    'f3b': {
        'bags': 2,
        'epochs': 7,
        'dataset': "f3",
    },
}


parser = argparse.ArgumentParser(description='Train FFM2 model')
parser.add_argument('profile', type=str, help='Train profile')
parser.add_argument('--rewrite-cache', action='store_true', help='Drop cache files prior to train')
parser.add_argument('--continue-train', type=str, help='Continue training of interrupted model')

args = parser.parse_args()

profile_name = args.profile
profile = profiles[profile_name]


if not os.path.exists('cache/full_train_bin_%s.index' % profile['dataset']) or args.rewrite_cache:
    print "Generating data..."
    os.system("bin/export-bin-data-%s" % profile['dataset'])


train_model(fit_predict, 'ffm2-%s' % profile_name, profile, name=args.continue_train)
