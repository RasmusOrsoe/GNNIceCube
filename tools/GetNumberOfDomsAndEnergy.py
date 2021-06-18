# -*- coding: utf-8 -*-
import pandas as pd
import sqlalchemy
import os
import sqlite3
import time
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import multiprocessing

def GetDomsAndEnergy(settings):
    events,save_path,id,db_file = settings
    n_doms = []
    energy = []
    with sqlite3.connect(db_file) as con:
       query = 'select event_no from features WHERE event_no IN %s'%(str(tuple(events)))
       features = pd.read_sql(query, con)
       query = 'select event_no,energy_log10 from truth WHERE event_no IN %s'%(str(tuple(events)))
       truth = pd.read_sql(query, con)
    count = 1
    for event in events:
        print('%s / %s'%(count, len(events)))
        n_doms.append(len((np.where(features['event_no'] == event))[0]))
        energy.append(float(truth['energy_log10'][np.where(truth['event_no'] == event)[0]]))
        count += 1
    results = pd.concat((pd.DataFrame(n_doms),pd.DataFrame(energy)),axis = 1)
    results.to_csv(save_path+'\\_%s.csv'%id,index = False)

if __name__ == '__main__':        
    batches = np.arange(2,4048,50)
    n_reps  = 10 
    db_file = r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\data\dev_numu_train_l5_retro_001.db'
    events_pure = pd.read_csv(r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\events\events.csv')
    events = events_pure['event_no'][0:500000]
    
    
    
    n_workers = 4
    event_list = np.array_split(events,n_workers)
    settings = []
    for k in range(len(event_list)):
        settings.append(tuple([event_list[k],
                               r'X:\speciale\data\export\dom_counting',
                               k,
                               db_file]))
    
    p = Pool(processes = len(settings))
    async_result = p.map(GetDomsAndEnergy, settings)
    p.close()
    p.join()