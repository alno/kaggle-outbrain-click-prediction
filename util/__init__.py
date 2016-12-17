import pandas as pd

import datetime


def gen_prediction_name(model_name, score):
    return "%s-%s-%.5f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), model_name, score)


def gen_submission(pred):
    pred = pred.sort_values(['display_id', 'score'], ascending=[True, False])[['display_id', 'ad_id']]

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
    pred = pred.sort_values(['display_id', 'score'], ascending=[True, False])[['display_id', 'clicked']]

    cur_idx = None
    cur_rank = None

    score_sum = 0.0
    score_cnt = 0

    for t in pred.itertuples():
        if cur_idx == t.display_id:
            cur_rank += 1
        else:
            score_cnt += 1
            cur_idx = t.display_id
            cur_rank = 1

        if t.clicked == 1:
            score_sum += 1.0 / cur_rank

    return score_sum / score_cnt
