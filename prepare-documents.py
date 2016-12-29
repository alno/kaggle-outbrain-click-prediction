import pandas as pd
import numpy as np

import os

from util.meta import cache_dir, input_dir


def fix_date(d):
    if type(d) in [str, unicode] and d.startswith('30'):
        d = d.replace('30', '20', 1)
    elif type(d) in [str, unicode] and d.startswith('00'):
        d = d.replace('00', '20', 1)

    return d


def encode_feature(values):
    uniq = values.unique()
    mapping = dict(zip(uniq, range(1, len(uniq) + 1)))

    return values.map(mapping)


df = pd.read_csv(os.path.join(input_dir, 'documents_meta.csv.zip'), index_col='document_id', dtype={'document_id': np.uint32})
df['source_id'] = df['source_id'].fillna(-1).astype(np.int32)
df['publisher_id'] = df['publisher_id'].fillna(-1).astype(np.int16)
df['publish_time'] = pd.to_datetime(df['publish_time'].map(fix_date).replace('nan', np.nan), errors='coerce')
df['publish_timestamp'] = (df['publish_time'].astype(np.int64) // 1000000 - 1465876799998).clip(lower=-1000000000000)

df.to_csv(os.path.join(cache_dir, 'documents.csv.gz'), compression='gzip')

## Document entities

df = pd.read_csv(os.path.join(input_dir, 'documents_entities.csv.zip'), index_col='document_id', dtype={'document_id': np.uint32})
df['entity_id'] = encode_feature(df['entity_id'])
df.to_csv(os.path.join(cache_dir, 'documents_entities.csv.gz'), compression='gzip')

print "Done."
