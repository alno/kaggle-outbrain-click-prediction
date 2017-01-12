import pandas as pd
import numpy as np

from util.meta import cv1_split, cv1_split_idx, cv1_split_time

print "Loading train..."

train = pd.read_csv("../input/clicks_train.csv.gz", dtype=np.int32)
train.reset_index(inplace=True)
train.rename(columns={'index': 'idx'}, inplace=True)

print "Loading events..."

events = pd.read_csv("../input/events.csv.gz", dtype=np.int32, index_col=0, usecols=[0, 3])  # Load events
events = events.loc[events.index.intersection(train.index.unique())]  # Take only train events

print "Splitting events..."

# Select display_ids for val - consists of time-based and sampled parts
train_is_val = train['display_id'].isin(events[(events['timestamp'] >= cv1_split_time) | (events.index % 6 == 5)].index)

del events

print "Splitting clicks..."

train.loc[~train_is_val, ['idx']].to_csv(cv1_split_idx[0], index=False, compression='gzip')
train.loc[~train_is_val, ['display_id', 'ad_id', 'clicked']].to_csv(cv1_split[0], index=False, compression='gzip')

train.loc[train_is_val, ['idx']].to_csv(cv1_split_idx[1], index=False, compression='gzip')
train.loc[train_is_val, ['display_id', 'ad_id', 'clicked']].to_csv(cv1_split[1], index=False, compression='gzip')

del train

print "Done."
