# -*- coding: utf-8 -*-
"""
This script will find all documents clicked historically by a user. 
However for extra granularity we store spearately whether the traffic source of the doucment
was 'internal', 'social' or search. For each, it is stored in a separate FFM namespace.    
We only store documents which occured at least 100 times in the events file. 
The script shuld be executed with pypy for faster execution. 
Code leveraged from rcarson @ https://www.kaggle.com/jiweiliu/outbrain-click-prediction/extract-leak-in-30-mins-with-small-memory
@author: darragh
"""
import csv, os, gzip

# define distionaries 
uuid_ev = {} # a check for if the user exists in events
ctdoc = {}   # check if the document occured at least 100 times. 

for c,row in enumerate(csv.DictReader(open('data/events.csv'))):
    if row['uuid'] != '':
        uuid_ev[row['uuid']] = 1 
    if row['document_id'] not in ctdoc:
        ctdoc[row['document_id']] = 1 
    else:
        ctdoc[row['document_id']] += 1 
print('all docs : ' + str(len(ctdoc)))
ctdoc = { key:value for key, value in ctdoc.items() if value > 100 }
print('common docs > 100 : ' + str(len(ctdoc)))

count = 0
outfile  = "uuid_doc_trfc_srce_ids.csv.gz"
filename = 'input/page_views.csv'
# filename = 'input/page_views_sample.csv' # comment this out locally

for c,row in enumerate(csv.DictReader(open(filename))):
    if c%1000000 == 0:
        print (c,count)
    if row['document_id'] not in ctdoc:
	    continue
    if row['uuid'] not in uuid_ev:
	    continue
    if uuid_ev[row['uuid']]==1:
	    uuid_ev[row['uuid']] = set()
    lu = len(uuid_ev[row['uuid']])
    uuid_ev[row['uuid']].add(str(int(row['traffic_source'])+6) + ':' + row['document_id'] + ':1')
    if lu!=len(uuid_ev[row['uuid']]):
	    count+=1

# Delete output file if it already exists
try:
    os.remove(outfile)
except OSError:
    pass

# Open the file to write to
fo = gzip.open(outfile,'w')
fo.write('uuid,document_id_trfc_srce\n')
for i in uuid_ev:
    if uuid_ev[i]!=1:
	    tmp = list(uuid_ev[i])
	    fo.write('%s,%s\n'%(i,' '.join(tmp)))
	    del tmp
fo.close()	
