import pandas as pd


def convert_id_list(doc_ids):
    return ' '.join(d.split(':')[1] for d in doc_ids.split(' '))


uids = pd.read_csv('cache/events.csv.gz', usecols=['uuid', 'uid'], index_col='uuid')['uid'].drop_duplicates()

dts = pd.read_csv('uuid_doc_trfc_srce_ids.csv')
dts['uid'] = dts['uuid'].map(uids)
dts['doc_trf_ids'] = dts['document_id_trfc_srce'].map(convert_id_list)

dts[['uid', 'doc_trf_ids']].to_csv('cache/viewed_doc_trf_source.csv.gz', index=False, compression='gzip')

print "Done."
