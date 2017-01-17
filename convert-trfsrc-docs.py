import pandas as pd


def convert_ids_filtered(field):
    def convert(doc_ids):
        triples = (d.split(':') for d in doc_ids.split(' '))

        return ' '.join(t[1] for t in triples if int(t[0]) == field)

    return convert


print "Loading uids..."
uids = pd.read_csv('cache/events.csv.gz', usecols=['uuid', 'uid'], index_col='uuid')['uid'].drop_duplicates()

print "Loading data..."
dts = pd.read_csv('uuid_doc_trfc_srce_ids.csv.gz')
dts['uid'] = dts['uuid'].map(uids)
dts['doc_src_int'] = dts['document_id_trfc_srce'].map(convert_ids_filtered(32))
dts['doc_src_soc'] = dts['document_id_trfc_srce'].map(convert_ids_filtered(33))
dts['doc_src_srh'] = dts['document_id_trfc_srce'].map(convert_ids_filtered(34))

print "Saving..."
dts[['uid', 'doc_src_int', 'doc_src_soc', 'doc_src_srh']].to_csv('cache/viewed_trfsrc_docs.csv.gz', index=False, compression='gzip')

print "Done."
