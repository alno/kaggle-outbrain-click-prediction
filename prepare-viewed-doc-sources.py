# -*- coding: utf-8 -*-
"""
This script will find the source (found in documents_meta) of each document clicked by a user.
This will be joined onto the training file with a key on the user.
This will be good to capture the sparse documents which would have been excluded from
other page_view features because the doucment was too sparse.
The script shuld be executed with pypy for faster execution.
@author: darragh
"""

import csv, os, gzip

input_dir = os.getenv('INPUT', '../input')

uuid_ev = {}  # a check for if the user exists in events
uuid_uid = {}  # Map of uuids to numeric ids
doc_srce = {}  # From documents_meta, the source of each document

for c, row in enumerate(csv.DictReader(gzip.open(input_dir + '/documents_meta.csv.gz'))):
    if row['source_id'] != '':
        doc_srce[row['document_id']] = row['source_id']

for c, row in enumerate(csv.DictReader(gzip.open('cache/events.csv.gz'))):
    if row['uuid'] != '':
        uuid_ev[row['uuid']] = 1
        uuid_uid[row['uuid']] = row['uid']

count = 0
outfile = "cache/viewed_doc_sources.csv.gz"
filename = input_dir + '/page_views.csv.gz'

# loop through the documents per user and get the source of the documents per user
for c, row in enumerate(csv.DictReader(gzip.open(filename))):
    if c % 1000000 == 0:
        print (c, count)

    if row['document_id'] not in doc_srce:
        continue
    if row['uuid'] not in uuid_ev:
        continue

    if uuid_ev[row['uuid']] == 1:
        uuid_ev[row['uuid']] = set()

    lu = len(uuid_ev[row['uuid']])
    uuid_ev[row['uuid']].add(doc_srce[row['document_id']])

    if lu != len(uuid_ev[row['uuid']]):
        count += 1

# Delete output file if it already exists
try:
    os.remove(outfile)
except OSError:
    pass

# Open the file to write to
fo = gzip.open(outfile, 'w')
fo.write('uuid,source_id\n')
for i in uuid_ev:
    if uuid_ev[i] != 1:
        tmp = list(uuid_ev[i])
        fo.write('%s,%s\n' % (uuid_uid[i], ' '.join(tmp)))
        del tmp
fo.close()
