import pandas as pd
import numpy as np

from util.meta import val_split, val_split_time

print "Loading train..."

train = pd.read_csv("../input/clicks_train.csv.gz", dtype=np.int32, index_col=0)

print "Loading events..."

events = pd.read_csv("../input/events.csv.gz", dtype=np.int32, index_col=0, usecols=[0, 3])  # Load events
events = events.loc[events.index.intersection(train.index.unique())]  # Take only train events

print "Splitting events..."

# Select display_ids for val - consists of time-based and sampled parts
train_is_val = train.index.isin(events[(events['timestamp'] >= val_split_time) | (events.index % 6 == 5)].index)

del events

print "Saving..."

train[~train_is_val].to_csv(val_split[0], index=True, compression='gzip')
train[train_is_val].to_csv(val_split[1], index=True, compression='gzip')

del train

print "Splitting leak..."

leak = pd.read_csv('cache/leak_full_train.csv.gz', dtype=np.uint8)
leak[~train_is_val].to_csv('cache/leak_val_train.csv.gz', index=False, compression='gzip')
leak[train_is_val].to_csv('cache/leak_val_test.csv.gz', index=False, compression='gzip')

del leak

print "Done."
