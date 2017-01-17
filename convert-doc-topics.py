import pandas as pd


def convert_id_list(doc_ids):
    return ' '.join(d.split(':')[1] for d in doc_ids.split(' '))


print "Loading uids..."
uids = pd.read_csv('cache/events.csv.gz', usecols=['uuid', 'uid'], index_col='uuid')['uid'].drop_duplicates()

print "Loading data..."
dts = pd.read_csv('uuid_topc_ids.csv.zip')
dts['uid'] = dts['uuid'].map(uids)
dts['topic_ids'] = dts['document_topc_id'].map(convert_id_list)

dts[['uid', 'topic_ids']].to_csv('cache/viewed_doc_topics.csv.gz', index=False, compression='gzip')

print "Done."
