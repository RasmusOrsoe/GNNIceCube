# -*- coding: utf-8 -*-
import pandas as pd
import sqlalchemy
import os
import sqlite3
import time
import matplotlib.pyplot as plt
import numpy as np

batches = np.arange(2,4048,50)
n_reps  = 10 
db_file = r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\data\dev_numu_train_l5_retro_001.db'
events_pure = pd.read_csv(r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\events\events.csv')
for j in range(0,n_reps):
    print('%s / %s'%(j,n_reps))
    timings = []
    for k in range(0,len(batches)):
        events = list(events_pure['event_no'][0:batches[k]])
        time_start = time.time()
        with sqlite3.connect(db_file) as con:
           query = 'select * from features WHERE event_no IN %s'%(str(tuple(events)))
           data_batch = pd.read_sql(query, con)
           query = 'select * from truth WHERE event_no IN %s'%(str(tuple(events)))
           data_batch = pd.read_sql(query, con)
        timings.append(time.time() - time_start)
    if j == 0:
        results = pd.DataFrame(timings)
    else:
        results = pd.concat((results,pd.DataFrame(timings)),axis = 1)
        
#%%

plt.scatter(batches,results.mean(axis = 1)/np.array(batches)*1000,linestyle = ':')
plt.title('Extraction Time pr. Event vs Batch Size',size  = 15)
plt.xlabel('Batch Size', size =  15)
plt.ylabel('Extraction Time pr. Event [$\mu s$]', size  = 15)

