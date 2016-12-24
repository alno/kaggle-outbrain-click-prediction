import pandas as pd
import numpy as np

import os

from util.meta import cache_dir, input_dir

df = pd.read_csv(os.path.join(input_dir, 'events.csv.zip'), index_col='display_id', dtype={'document_id': np.uint32})
df['platform'] = df['platform'].replace({'\N': 0}).astype(np.uint8)

location = df['geo_location'].fillna('').str.split('>')
df.drop(['geo_location'], axis=1, inplace=True)

df['country'] = location.map(lambda loc: loc[0] if len(loc) > 0 else 'Z')
df['state'] = location.map(lambda loc: loc[1] if len(loc) > 1 else 'Z')
df['region'] = location.map(lambda loc: int(loc[2]) if len(loc) > 2 else -1)

df.to_csv(os.path.join(cache_dir, 'events.csv.gz'), compression='gzip')

print "Done."
