# -*- coding: utf-8 -*-
"""
This script will find the documents clicked for a user within one hour after an 
event has occurred. It works using the events file where we have the user and 
the timestamp and we go to page_views and check which clicks are within one
hour of than timestamp for the user. 
This is a form of leak, and most likely not useful in production, as we use 
future information of the event to identify the ad picked.
The script shuld be executed with pypy for faster execution. 
@author: darragh
"""

import csv, os, gzip

uuidtstamps = {}   # store the users along with the ties they were presented ads
uuidhrafter = {}   # store the folloing infomration from each event 
                   # key:{user, event timestamps} and value{all documents clicked within one hour}
ctdoc = {}         # the count of how often docs appear

# Get timestamp user combination for each events
for c,row in enumerate(csv.DictReader(open('input/events.csv'))):
    if row['uuid'] not in uuidtstamps:
        uuidtstamps[row['uuid']] = set()
    uuidtstamps[row['uuid']].add(row['timestamp'])
    uuidhrafter [row['uuid']+'_'+row['timestamp']] = 1 


count = 0
outfile  = "uuid_onehour_after.csv.gz"
filename = 'input/page_views.csv'
# filename = 'input/page_views_sample.csv' # comment this out locally

# Count documents which occured less than 80 times and exclude them 
for c,row in enumerate(csv.DictReader(open(filename))):
    if row['uuid'] not in uuidtstamps:
	    continue
    if row['document_id'] not in ctdoc:
        ctdoc[row['document_id']] = 1 
    else:
        ctdoc[row['document_id']] += 1 
print('all docs : ' + str(len(ctdoc)))
ctdoc = { key:value for key, value in ctdoc.items() if value > 80 }
print('common docs > 80 : ' + str(len(ctdoc)))

# for each page_views row where we get a uuid match, and the document occurs over 
# the required 80 count, we loop through the users click timestamps to find if 
# is within one hour of any of the event timestamps. 
for c,row in enumerate(csv.DictReader(open(filename))):
    if c%1000000 == 0:
        print (c,count)
    if row['document_id'] not in ctdoc:
	    continue
    if row['uuid'] not in uuidtstamps:
	    continue
    for time in uuidtstamps[row['uuid']]:
        diff = int(row['timestamp']) - int(time) 
        if abs(diff) < 3600*1000:
            if diff > 0:
                if uuidhrafter[row['uuid']+'_'+ time] == 1:
                    uuidhrafter[row['uuid']+'_'+ time] = set()
                uuidhrafter[row['uuid']+'_'+ time].add('10:'+row['document_id']+':1')
        del diff
     
# Delete output file if it already exists
try:
    os.remove(outfile)
except OSError:
    pass

# Open the file to write to
fo = gzip.open(outfile,'w')
fo.write('uuid, timestamp, document_id\n')
for i in uuidhrafter:
    if uuidhrafter[i]!=1:
        tmp = list(uuidhrafter[i])
        utime = i.split('_')
        fo.write('%s,%s,%s\n'%(utime[0], utime[1],' '.join(tmp)))
        del tmp, utime
fo.close()	