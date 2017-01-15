import pandas as pd
import numpy as np

from itertools import izip
from tqdm import tqdm

from scipy.special import logit

from util.meta import row_counts


chunk_size = 100000
ctr_smooth = 10


def ctr_logit(views, clicks):
    return logit((clicks + 0.194 * ctr_smooth) / (views + ctr_smooth))


def export_data(clicks_file_name, out_name):
    n_rows = row_counts[out_name]
    res = np.memmap("cache/%s_np_ctr1.npy" % out_name, dtype='float32', mode='w+', shape=(n_rows, 19))

    click_stream = pd.read_csv(clicks_file_name, dtype=np.uint32, chunksize=chunk_size)
    leak_stream = pd.read_csv("cache/leak_%s.csv.gz" % out_name, dtype=np.uint16, chunksize=chunk_size)
    rival_stream = pd.read_csv("cache/rivals_%s.csv.gz" % out_name, usecols=['rival_count'], dtype=np.uint16, chunksize=chunk_size)

    uid_viewed_ads_stream = pd.read_csv("cache/uid_viewed_ads_%s.csv.gz" % out_name, dtype=np.uint16, chunksize=chunk_size)
    uid_viewed_ad_srcs_stream = pd.read_csv("cache/uid_viewed_ad_srcs_%s.csv.gz" % out_name, dtype=np.uint16, chunksize=chunk_size)
    uid_viewed_ad_cats_stream = pd.read_csv("cache/uid_viewed_ad_cats_%s.csv.gz" % out_name, dtype=np.float32, chunksize=chunk_size)
    uid_viewed_ad_tops_stream = pd.read_csv("cache/uid_viewed_ad_tops_%s.csv.gz" % out_name, dtype=np.float32, chunksize=chunk_size)

    g2_viewed_ads_stream = pd.read_csv("cache/g2_viewed_ads_%s.csv.gz" % out_name, dtype=np.uint16, chunksize=chunk_size)
    g2_viewed_ad_srcs_stream = pd.read_csv("cache/g2_viewed_ad_srcs_%s.csv.gz" % out_name, dtype=np.uint16, chunksize=chunk_size)
    g2_viewed_ad_cats_stream = pd.read_csv("cache/g2_viewed_ad_cats_%s.csv.gz" % out_name, dtype=np.float32, chunksize=chunk_size)
    #g2_viewed_ad_tops_stream = pd.read_csv("cache/g2_viewed_ad_tops_%s.csv.gz" % out_name, dtype=np.float32, chunksize=chunk_size)

    zipped_stream = izip(
        click_stream, leak_stream, rival_stream,
        uid_viewed_ads_stream, uid_viewed_ad_srcs_stream, uid_viewed_ad_cats_stream, uid_viewed_ad_tops_stream,
        g2_viewed_ads_stream, g2_viewed_ad_srcs_stream, g2_viewed_ad_cats_stream
    )

    chunk_start = 0
    with tqdm(total=n_rows, desc='  Exporting %s' % clicks_file_name, unit='rows') as pbar:
        for clk, leak, riv, uv_ad, uv_ad_src, uv_ad_cat, uv_ad_top, g2_ad, g2_ad_src, g2_ad_cat in zipped_stream:
            chunk_end = chunk_start + clk.shape[0]

            res[chunk_start:chunk_end, 0] = (leak['viewed'] > 0).astype(np.float32)
            res[chunk_start:chunk_end, 1] = (leak['not_viewed'] > 0).astype(np.float32)

            res[chunk_start:chunk_end, 2] = ctr_logit(uv_ad['ad_doc_past_views'], uv_ad['ad_doc_past_clicks'])
            res[chunk_start:chunk_end, 3] = ctr_logit(uv_ad['ad_doc_future_views'], uv_ad['ad_doc_future_clicks'])

            res[chunk_start:chunk_end, 4] = ctr_logit(uv_ad_src['src_past_views'], uv_ad_src['src_past_clicks'])
            res[chunk_start:chunk_end, 5] = ctr_logit(uv_ad_src['src_future_views'], uv_ad_src['src_future_clicks'])

            res[chunk_start:chunk_end, 6] = ctr_logit(uv_ad_top['top_past_views'], uv_ad_top['top_past_clicks'])
            res[chunk_start:chunk_end, 7] = ctr_logit(uv_ad_top['top_future_views'], uv_ad_top['top_future_clicks'])

            res[chunk_start:chunk_end, 8] = ctr_logit(uv_ad_cat['cat_past_views'], uv_ad_cat['cat_past_clicks'])
            res[chunk_start:chunk_end, 9] = ctr_logit(uv_ad_cat['cat_future_views'], uv_ad_cat['cat_future_clicks'])

            res[chunk_start:chunk_end, 10] = ctr_logit(g2_ad['ad_doc_past_views'], g2_ad['ad_doc_past_clicks'])
            res[chunk_start:chunk_end, 11] = ctr_logit(g2_ad['ad_doc_future_views'], g2_ad['ad_doc_future_clicks'])

            res[chunk_start:chunk_end, 12] = ctr_logit(g2_ad_src['src_past_views'], g2_ad_src['src_past_clicks'])
            res[chunk_start:chunk_end, 13] = ctr_logit(g2_ad_src['src_future_views'], g2_ad_src['src_future_clicks'])

            res[chunk_start:chunk_end, 14] = 0#ctr_logit(g2_ad_top['top_past_views'], g2_ad_top['top_past_clicks'])
            res[chunk_start:chunk_end, 15] = 0#ctr_logit(g2_ad_top['top_future_views'], g2_ad_top['top_future_clicks'])

            res[chunk_start:chunk_end, 16] = ctr_logit(g2_ad_cat['cat_past_views'], g2_ad_cat['cat_past_clicks'])
            res[chunk_start:chunk_end, 17] = ctr_logit(g2_ad_cat['cat_future_views'], g2_ad_cat['cat_future_clicks'])

            res[chunk_start:chunk_end, 18] = logit(1.0 / riv['rival_count'])

            chunk_start += clk.shape[0]
            pbar.update(clk.shape[0])


export_data('cache/clicks_cv2_train.csv.gz', 'cv2_train')
export_data('cache/clicks_cv2_test.csv.gz', 'cv2_test')

export_data('cache/clicks_cv1_train.csv.gz', 'cv1_train')
export_data('cache/clicks_cv1_test.csv.gz', 'cv1_test')

export_data('../input/clicks_train.csv.gz', 'full_train')
export_data('../input/clicks_test.csv.gz', 'full_test')

print "Done."
