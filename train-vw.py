import numpy as np
import pandas as pd

import os
import time
import argparse

from scipy.special import expit

from util.meta import full_split, val_split
from util.data import read_events, read_documents_meta, read_promoted_content
from util import gen_prediction_name, gen_submission, score_prediction


parser = argparse.ArgumentParser(description='Train VW model')
parser.add_argument('--rewrite-cache', action='store_true', help='Drop cache files prior to train')


args = parser.parse_args()

print "Loading dictionary data..."

events = read_events()
documents = read_documents_meta()
promoted_content = read_promoted_content()


def export_data(src_file, dst_file, label=False):
    if os.path.exists(dst_file) and not args.rewrite_cache:
        return

    print "  Writing %s..." % dst_file

    start_time = time.time()

    with open(dst_file, 'wb', buffering=256*1024) as f:
        for df in pd.read_csv(src_file, dtype=np.int32, chunksize=2000000):
            idx = df.index  # Save record order

            df = df.merge(events, left_on='display_id', right_index=True)
            df = df.merge(promoted_content, left_on='ad_id', right_index=True)
            df = df.loc[idx]  # Restore correct order

            # TODO Join other data

            if label:
                target = df['clicked'].values * 2 - 1

            ad_id = df['ad_id'].values
            ad_document_id = df['ad_document_id'].values
            document_id = df['document_id'].values
            platform = df['platform'].values
            location = df['geo_location'].fillna('').str.split('>')

            for i in xrange(df.shape[0]):
                line = ''

                if label:
                    line += '%f ' % target[i]

                line += '|a ad_%d l_c_%s p_%d' % (ad_id[i], location[i][0] if len(location[i]) > 0 else 'UNK', platform[i])
                line += '|d ad_d_%d d_%d' % (ad_document_id[i], document_id[i])

                f.write(line + '\n')

    print "    Completed in %d seconds" % (time.time() - start_time)


def fit_predict(split, split_name):
    train_file = 'cache/%s_train_vw.txt' % split_name
    pred_file = 'cache/%s_test_vw.txt' % split_name

    export_data(split[0], train_file, label=True)
    export_data(split[1], pred_file)

    print "  Training..."

    if os.path.exists(train_file + '.cache'):
        os.remove(train_file + '.cache')

    os.system("vw --cache --passes 3 -P 5000000 --loss_function logistic -q aa -q dd -f vw.model %s " % train_file)

    print "  Predicting..."

    if os.path.exists(pred_file + '.cache'):
        os.remove(pred_file + '.cache')

    os.system("vw -i vw.model -p vw.preds -P 5000000 -t %s" % pred_file)

    pred = pd.read_csv(split[1])
    pred['pred'] = expit(np.loadtxt('vw.preds'))

    return pred

##

print "Validation split..."

pred = fit_predict(val_split, 'val')

print "Scoring..."

present_score, future_score, score = score_prediction(pred)
name = gen_prediction_name('vw', score)

print "  Present score: %.5f" % present_score
print "  Future score: %.5f" % future_score
print "  Total score: %.5f" % score

pred[['display_id', 'ad_id', 'pred']].to_pickle('preds/%s-val.pickle' % name)

##

print "Full split..."

pred = fit_predict(full_split, 'full')
pred[['display_id', 'ad_id', 'pred']].to_pickle('preds/%s-test.pickle' % name)

print "  Generating submission..."
subm = gen_submission(pred)
subm.to_csv('subm/%s.csv.gz' % name, index=False, compression='gzip')

print "  File name: %s" % name
print "Done."
