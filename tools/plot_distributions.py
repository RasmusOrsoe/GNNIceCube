import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt

def grabfiles(path):
    files =  os.listdir(path)
    events = pd.DataFrame()
    for file in files:
        events = events.append(pd.read_csv(path + '//'+file))
    return events



train_path = r'X:\speciale\data\graphs\dev_numu_train_l5_retro_001\event_only_2mio_retro_SRT\train'
valid_path = r'X:\speciale\data\graphs\dev_numu_train_l5_retro_001\event_only_2mio_retro_SRT\valid'

train_events = grabfiles(train_path)['event_no']
valid_events = grabfiles(valid_path)['event_no']

mc_db = r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\data\dev_numu_train_l5_retro_001.db'
with sqlite3.connect(mc_db) as con:
    query = 'select * from sca limit 1'
    truth = pd.read_sql(query,con)
    query = 'select * from features limit 1'
    feat = pd.read_sql(query,con)


feats    = feat.columns

truths  = truth.columns
no_plot = ['event_no','pid','stopped_muon','muon_track_length']
for key in truth:
    if key not in no_plot:
        if 'sigma' not in key:
            if 'retro' not in key:
                with sqlite3.connect(mc_db) as con:
                    query = 'select %s from truth where event_no in %s'%(key, str(tuple(valid_events)))
                    truth_valid = pd.read_sql(query,con)
                    query = 'select %s from truth where event_no in %s'%(key, str(tuple(train_events)))
                    truth_train = pd.read_sql(query,con)
            
                fig = plt.figure()
                plt.title(key,size = 30)
                plt.hist(truth_train[key],bins = 60, histtype = 'step',label = 'Training Sample')
                plt.hist(truth_valid[key],bins = 60, histtype = 'step',label = 'Validation Sample')
                plt.legend(fontsize = 25)
                plt.xlabel('Transformed %s'%key,size  = 25)
                plt.xticks(size = 20)
                plt.yticks(size = 20)
                
