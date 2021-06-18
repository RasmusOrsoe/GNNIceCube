# -*- coding: utf-8 -*-
import pandas as pd
import sqlalchemy
import os
import sqlite3
import time
import matplotlib.pyplot as plt
import numpy as np


db_file = r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\data\dev_numu_train_l5_retro_001.db'
events_pure = pd.read_csv(r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\events\events.csv')

events = list(events_pure['event_no'][0:100000])
time_start = time.time()
with sqlite3.connect(db_file) as con:
   query = 'select x from features WHERE event_no IN %s '%(str(tuple(events)))
   data_batch = pd.read_sql(query, con)

with sqlite3.connect(db_file) as con:
   query = 'select x from features WHERE event_no IN %s and SRTInIcePulses = 1'%(str(tuple(events)))
   data_batch_SRT = pd.read_sql(query, con)





#%%
scalers =  pd.read_pickle(r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\meta\transformers.pkl')
fig, axs = plt.subplots(1,3)

key = 'zenith'
scaler  = scalers['truth'][key] 
axs[0].hist(bins = 50, x = scaler.inverse_transform(np.array(data_batch[key]).reshape((-1,1))))
axs[0].set_title('%s'%key,size  = 15)
axs[0].set_xlabel('Degrees [rad]', size =  15)
axs[0].set_ylabel('Count', size  = 15)

key = 'energy_log10'
scaler  = scalers['truth'][key] 
axs[1].hist(bins = 50, x = scaler.inverse_transform(np.array(data_batch[key]).reshape((-1,1))))
axs[1].set_title('%s'%key,size  = 15)
axs[1].set_xlabel('Energy [GeV]', size =  15)
axs[1].set_ylabel('Count', size  = 15)

key = 'azimuth'
scaler  = scalers['truth'][key] 
axs[2].hist(bins = 50, x = scaler.inverse_transform(np.array(data_batch[key]).reshape((-1,1))))
axs[2].set_title('%s'%key,size  = 15)
axs[2].set_xlabel('Degrees [rad]', size =  15)
axs[2].set_ylabel('Count', size  = 15)

