import pandas as pd


def convert_document_id(doc_ids):
    return ' '.join(d.split(':')[1] for d in doc_ids.split(' '))

df = pd.read_csv("document_ad_id_others.csv.gz")
df['document_ad_id'] = df['document_ad_id'].map(convert_document_id)
df.to_csv('cache/doc_ad_others.csv.gz', index=False, compression='gzip')

print "Done."
