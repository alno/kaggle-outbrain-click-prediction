import pandas as pd


def convert_document_id(doc_ids):
    return ' '.join(d.split(':')[1] for d in doc_ids.split(' '))


uids = pd.read_csv('cache/events.csv.gz', usecols=['uuid', 'uid'], index_col='uuid')['uid'].drop_duplicates()

oha = pd.read_csv('uuid_onehour_after.csv')
oha['uid'] = oha['uuid'].map(uids)
oha['doc_ids'] = oha['document_id'].map(convert_document_id)

oha[['uid', 'timestamp', 'doc_ids']].to_csv('cache/viewed_docs_one_hour_after.csv.gz', index=False, compression='gzip')

print "Done."
