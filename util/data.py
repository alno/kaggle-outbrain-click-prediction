import pandas as pd
import numpy as np

import os


from .meta import input_dir, cache_dir


def read_events():
    return pd.read_csv(os.path.join(cache_dir, 'events.csv.gz'), index_col='display_id', dtype={'document_id': np.uint32, 'platform': np.uint8, 'region': np.uint32})


def read_documents():
    return pd.read_csv(os.path.join(cache_dir, 'documents.csv.gz'), index_col='document_id', dtype={'document_id': np.uint32, 'source_id': np.int32, 'publisher_id': np.int16}, parse_dates=['publish_time'])


def read_ads():
    df = pd.read_csv(os.path.join(input_dir, 'promoted_content.csv.gz'), index_col='ad_id', dtype={'ad_id': np.uint32, 'document_id': np.uint32, 'campaign_id': np.uint16, 'advertiser_id': np.uint16})
    df.rename(columns={'document_id': 'ad_document_id'}, inplace=True)

    return df
