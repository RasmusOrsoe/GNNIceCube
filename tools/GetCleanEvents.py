import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from multiprocessing import Pool
import multiprocessing
import time


def GetMetaInfo(settings):
    
    db_file,events,scaler_path,worker_id,save_path = settings
    
    scaler = pd.read_pickle(scaler_path)['truth']['energy_log10']
    dom20 = list()
    dom110 = list()
    dom120 = list()
    dom130 = list()
    event_list = tuple(events)
    count = 0
    cut_events  = list()
    for i in range(0,len(event_list)):
        if count == 1000:
            count = 0
            print('%s / %s'%(i,len(event_list))) 
        with sqlite3.connect(db_file) as con:         
            query = 'select pmt_type from features WHERE event_no = %s and SRTInIcePulses = 1'%(str(event_list[i]))                                              # THESE ARE THEN WRITTEN TO DRIVE
            sca = pd.read_sql(query, con)                             #
            cursor = con.cursor()
        count +=1
        dom20 = sum(sca['pmt_type'] == 20)
        dom110 = sum(sca['pmt_type'] == 110)
        dom120 = sum(sca['pmt_type'] == 120)
        dom130 = sum(sca['pmt_type'] == 130)
        if(dom20 >= 4):
            if(dom120 + dom130 >= 1):
                cut_events.append(event_list[i])
    
    data = pd.DataFrame(cut_events)
    data.columns = ['event_no']
    data.to_csv(save_path + '\\' + '%s'%worker_id + '_event_cutoff.csv')
    return

####################################################
# SETTINGS
####################################################
if __name__ == '__main__':
    path = r'X:\speciale\data\raw\dev_upgrade_train_step4_001\clean_events\all_cleaned.csv'
    data_handle     = 'dev_upgrade_train_step4_001'
    db_file         = r'X:\speciale\data\raw\%s\data\%s.db'%(data_handle,data_handle)
    events          = pd.read_csv(r'X:\speciale\data\raw\dev_upgrade_train_step4_001\events\events.csv').loc[:,'event_no'].reset_index(drop=True)[0:3000000]
    scaler_path     = r'X:\speciale\data\raw\%s\meta\transformers.pkl'%data_handle
    n_workers       = 5
    save_path       = r'X:\speciale\data\export\Upgrade Cleaning\distributions\automated'
    
    settings = list()
    event_list = np.array_split(events,n_workers)
    for k in range(len(event_list)):
        settings.append(tuple([db_file,event_list[k],scaler_path,k,save_path]))
    start = time.time()
    #GetMetaInfo(settings[0])
    p = Pool(processes = len(settings))
    async_result = p.map_async(GetMetaInfo, settings)
    p.close()
    p.join()
    print("Complete")
    end = time.time()
    print('total time (s)= ' + str(end-start))