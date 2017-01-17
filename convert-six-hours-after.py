import pandas as pd


def convert_id_list(doc_ids):
    return ' '.join(d.split(':')[1] for d in doc_ids.split(' '))


print "Loading uids..."
uids = pd.read_csv('cache/events.csv.gz', usecols=['uuid', 'uid'], index_col='uuid')['uid'].drop_duplicates()

print "Loading data..."
oha = pd.read_csv('uuid_siz_hours_after.csv.zip')
oha.rename(columns={' timestamp': 'timestamp', ' document_id': 'document_id'}, inplace=True)
oha['uid'] = oha['uuid'].map(uids)
oha['doc_ids'] = oha['document_id'].map(convert_id_list)

print "Saving..."
oha[['uid', 'timestamp', 'doc_ids']].to_csv('cache/viewed_docs_six_hours_after.csv.gz', index=False, compression='gzip')

print "Done."
