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

uuid_ev = {} # a check for if the user exists in events
doc_srce = {} # From documents_meta, the source of each document

for c,row in enumerate(csv.DictReader(open('data/documents_meta.csv'))):
    if row['source_id'] != '':
        doc_srce[row['document_id']] = row['source_id'] 
for c,row in enumerate(csv.DictReader(open('data/events.csv'))):
    if row['uuid'] != '':
        uuid_ev[row['uuid']] = 1 

count = 0
outfile  = "uuid_srce_ids.csv.gz"
filename = 'input/page_views.csv'
# filename = 'input/page_views_sample.csv' # comment this out locally

# loop through the documents per user and get the source of the documents per user
for c,row in enumerate(csv.DictReader(open(filename))):
    if c%1000000 == 0:
        print (c,count)
    if row['document_id'] not in doc_srce:
	    continue
    if row['uuid'] not in uuid_ev:
	    continue
    if uuid_ev[row['uuid']]==1:
	    uuid_ev[row['uuid']] = set()
    lu = len(uuid_ev[row['uuid']])
    uuid_ev[row['uuid']].add('6:' + doc_srce[row['document_id']] + ':1')
    if lu!=len(uuid_ev[row['uuid']]):
	    count+=1
     
# Delete output file if it already exists
try:
    os.remove(outfile)
except OSError:
    pass

# Open the file to write to
fo = gzip.open(outfile,'w')
fo.write('uuid,document_id_source\n')
for i in uuid_ev:
    if uuid_ev[i]!=1:
	    tmp = list(uuid_ev[i])
	    fo.write('%s,%s\n'%(i,' '.join(tmp)))
	    del tmp
fo.close()	