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


parser = argparse.ArgumentParser(description='Train FFM2 model')
parser.add_argument('--rewrite-cache', action='store_true', help='Drop cache files prior to train')
parser.add_argument('--continue-train', type=str, help='Continue training of interrupted model')

args = parser.parse_args()

profile_name = 'ffm2-f4b'
profile = {
    'bags': 2,
    'epochs': 7,
    'dataset': "f4",
}


if not os.path.exists('cache/full_train_bin_%s.index' % profile['dataset']) or args.rewrite_cache:
    print "Generating data..."
    os.system("bin/export-bin-data-%s" % profile['dataset'])


train_model(fit_predict, profile_name, profile, name=args.continue_train)
