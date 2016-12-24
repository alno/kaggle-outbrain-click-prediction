import pandas as pd

from util.meta import full_split, val_split
from util import gen_prediction_name, gen_submission, score_prediction

preds = [
    ('20161225-0051-ffm-0.65640', 0.7),
    ('20161224-2245-vw-0.64495', 0.3)
]


def fit_predict(split, split_name):
    pred = pd.read_csv(split[1])
    pred['pred'] = sum(pd.read_pickle('preds/%s-%s.pickle' % (p, 'test' if split_name == 'full' else 'val'))['pred'] * w for p, w in preds)

    return pred


## Validation

print "Validation split..."

pred = fit_predict(val_split, 'val')

print "  Scoring..."

present_score, future_score, score = score_prediction(pred)
name = gen_prediction_name('l2-blend', score)

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
