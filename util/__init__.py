import pandas as pd
import numpy as np

import datetime
import os

from numba import jit

from .meta import cv1_split_time, full_split, cv1_split, cv2_split


def gen_prediction_name(model_name, score):
    return "%s-%s-%.5f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), model_name, score)


def gen_submission(pred):
    pred = pred.sort_values(['display_id', 'pred'], ascending=[True, False])[['display_id', 'ad_id']]

    res_idx = []
    res_ads = []

    for t in pred.itertuples():
        if len(res_idx) > 0 and res_idx[-1] == t.display_id:
            if len(res_ads[-1]) < 12:
                res_ads[-1].append(str(t.ad_id))
        else:
            res_idx.append(t.display_id)
            res_ads.append([str(t.ad_id)])

    return pd.DataFrame({'display_id': res_idx, 'ad_id': [' '.join(a) for a in res_ads]})


def score_prediction(pred):
    pred = pred.sort_values(['display_id', 'pred'], ascending=[True, False])[['display_id', 'clicked']]
    pred = pd.merge(pred, pd.read_csv("cache/events.csv.gz", dtype=np.int32, index_col=0, usecols=[0, 3]), left_on='display_id', right_index=True)

    cur_idx = None
    cur_rank = None

    future_score_sum = 0.0
    future_score_cnt = 0

    present_score_sum = 0.0
    present_score_cnt = 0

    for t in pred.itertuples():
        if cur_idx == t.display_id:
            cur_rank += 1
        else:
            if t.timestamp >= cv1_split_time:
                future_score_cnt += 1
            else:
                present_score_cnt += 1

            cur_idx = t.display_id
            cur_rank = 1

        if t.clicked == 1:
            if t.timestamp >= cv1_split_time:
                future_score_sum += 1.0 / cur_rank
            else:
                present_score_sum += 1.0 / cur_rank

    present_score = present_score_sum / present_score_cnt
    future_score = future_score_sum / future_score_cnt
    total_score = (present_score_sum + future_score_sum) / (present_score_cnt + future_score_cnt)

    return present_score, future_score, total_score


def print_and_exec(cmd):
    print cmd
    os.system(cmd)


@jit
def score_sorted(y_true, y_pred, y_group):
    cur_group = -1

    start_idx = -1
    true_idx = -1

    score_sum = 0.0
    score_cnt = 0

    for i in xrange(len(y_true)):
        if y_group[i] > cur_group:
            if cur_group >= 0:
                rank = 0

                for j in xrange(start_idx, i):
                    if y_pred[j] >= y_pred[true_idx]:
                        rank += 1

                if rank > 0 and rank <= 12:
                    score_sum += 1.0 / rank

                score_cnt += 1

            start_idx = i
            true_idx = -1
            cur_group = y_group[i]

        if y_true[i]:
            true_idx = i

    return score_sum / score_cnt


def train_model(fit_predict, model_name, profile, name=None):

    ## Validation on CV2
    if name is not None and os.path.exists('preds/%s-cv2.csv.gz' % name):
        print "CV2 results already exist, skipping..."
    else:
        print "CV2 split..."

        pred = fit_predict(profile, cv2_split, 'cv2')

        print "  Scoring..."

        cv2_present_score, cv2_future_score, cv2_score = score_prediction(pred)

        if name is None:
            name = gen_prediction_name(model_name, cv2_score)

        print "  Present score: %.5f" % cv2_present_score
        print "  Future score: %.5f" % cv2_future_score
        print "  Total score: %.5f" % cv2_score

        pred[['pred']].to_csv('preds/%s-cv2.csv.gz' % name, index=False, compression='gzip')

        del pred

    ## Validation on CV1
    if os.path.exists('preds/%s-cv1.csv.gz' % name):
        print "CV1 results already exist, skipping..."
    else:
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
    if os.path.exists('preds/%s-test.csv.gz' % name):
        print "Full results already exist, skipping..."
    else:
        print "Full split..."

        pred = fit_predict(profile, full_split, 'full')
        pred[['pred']].to_csv('preds/%s-test.csv.gz' % name, index=False, compression='gzip')

        print "  Generating submission..."
        subm = gen_submission(pred)
        subm.to_csv('subm/%s.csv.gz' % name, index=False, compression='gzip')

        del pred, subm

        print "  File name: %s" % name

    print "Done."
