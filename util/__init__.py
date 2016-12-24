import pandas as pd
import numpy as np

import datetime
import os

from .meta import val_split_time


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
    pred = pd.merge(pred, pd.read_csv("../input/events.csv.zip", dtype=np.int32, index_col=0, usecols=[0, 3]), left_on='display_id', right_index=True)

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
            if t.timestamp >= val_split_time:
                future_score_cnt += 1
            else:
                present_score_cnt += 1

            cur_idx = t.display_id
            cur_rank = 1

        if t.clicked == 1:
            if t.timestamp >= val_split_time:
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
