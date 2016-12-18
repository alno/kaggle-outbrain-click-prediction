import pandas as pd
import numpy as np

import os


from .meta import input_dir


def read_events():
    df = pd.read_csv(os.path.join(input_dir, 'events.csv.zip'), index_col='display_id', dtype={'document_id': np.uint32})
    df['platform'] = df['platform'].replace({'\N': 0}).astype(np.uint8)

    return df


def read_promoted_content():
    df = pd.read_csv(os.path.join(input_dir, 'promoted_content.csv.zip'), index_col='ad_id', dtype={'ad_id': np.uint32, 'document_id': np.uint32, 'campaign_id': np.uint16, 'advertiser_id': np.uint16})
    df.rename(columns={'document_id': 'ad_document_id'}, inplace=True)

    return df


def read_documents_meta():
    df = pd.read_csv(os.path.join(input_dir, 'documents_meta.csv.zip'), index_col='document_id', dtype={'document_id': np.uint32}, parse_dates=['publish_time'])
    df['source_id'] = df['source_id'].fillna(-1).astype(np.int32)
    df['publisher_id'] = df['publisher_id'].fillna(-1).astype(np.int16)

    return df
