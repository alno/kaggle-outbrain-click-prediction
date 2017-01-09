import pandas as pd

from util.meta import full_split, val_split
from util import gen_prediction_name, gen_submission, score_prediction

preds = [
    ('20170108-2008-ffm2-f1-0.68984', 1.3),

    ('20170107-2248-ffm2-p1-0.68876', 0.9),
    ('20170108-0345-ffm2-p2-0.68762', 0.7),

    ('20161230-1323-ffm-p1-0.68204', 0.2),
    ('20161230-1049-ffm-p2-0.68169', 0.2),

    ('20161231-0544-vw-p1-0.67309', 0.07),
    ('20161231-1927-vw-p2-0.66718', 0.03),

    ('20170106-1339-vw-p1-0.67829', 0.1),
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

pred[['pred']].to_pickle('preds/%s-val.pickle' % name)

del pred

## Prediction

print "Full split..."

pred = fit_predict(full_split, 'full')
pred[['pred']].to_pickle('preds/%s-test.pickle' % name)

print "  Generating submission..."
subm = gen_submission(pred)
subm.to_csv('subm/%s.csv.gz' % name, index=False, compression='gzip')

del pred, subm

print "  File name: %s" % name
print "Done."
