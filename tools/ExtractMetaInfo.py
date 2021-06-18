import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from multiprocessing import Pool
import multiprocessing
import time
    

def GetMetaInfo(settings):
    
    db_file,events,scaler_path,worker_id,save_path = settings
    
    
    events  = events.reset_index(drop = True)
    scaler = pd.read_pickle(scaler_path)['truth']['energy_log10']
    dom20 = list()
    dom110 = list()
    dom120 = list()
    dom130 = list()
    n_strings = list()
    n_doms = list()
    energy = list()
    event_list = tuple(events)
    count = 0
    for i in range(0,len(event_list)):
        if count == 1000:
            count = 0
            print('%s / %s'%(i,len(event_list))) 
        with sqlite3.connect(db_file) as con:  
            query = 'select energy_log10 from truth WHERE event_no = %s'%(str(event_list[i]))
            E = pd.read_sql(query, con)          
            query = 'select pmt_type,string from features WHERE event_no = %s and SRTInIcePulses = 1 and RFilter = 1'%(str(event_list[i]))                                              # THESE ARE THEN WRITTEN TO DRIVE
            sca = pd.read_sql(query, con)                             #
            cursor = con.cursor()
        count +=1
        dom20.append(sum(sca['pmt_type'] == 20))
        dom110.append(sum(sca['pmt_type'] == 110))
        dom120.append(sum(sca['pmt_type'] == 120))
        dom130.append(sum(sca['pmt_type'] == 130))
        energy.append(np.array(E['energy_log10']))
        n_strings.append(len(pd.unique(sca['string'])))
    
    data = pd.concat([events.reset_index(drop=True),
                      pd.DataFrame(energy).reset_index(drop=True),
                      pd.DataFrame(dom20).reset_index(drop=True),
                      pd.DataFrame(dom110).reset_index(drop=True),
                      pd.DataFrame(dom120).reset_index(drop=True),
                      pd.DataFrame(dom130).reset_index(drop=True),
                      pd.DataFrame(n_strings).reset_index(drop=True)],axis = 1)
    
    data.columns = ['event_no','energy_log10','dom20','dom110','dom120','dom130','n_strings']
    data['energy_log10'] = scaler.inverse_transform(np.array(data.loc[:,'energy_log10']).reshape(-1,1))
    data.to_csv(save_path + '\\' + '%s'%worker_id + 'SRT_rfilter.csv', index = False)
    return

####################################################
# SETTINGS
####################################################
if __name__ == '__main__':
    path            = r'X:\speciale\data\raw\dev_upgrade_train_step4_001_martin_rfilter_000v3\events\events.csv' #r'X:\speciale\data\raw\dev_upgrade_train_step4_001\events\events.csv'
    data_handle     = 'dev_upgrade_train_step4_001_martin_rfilter_000v3'
    db_file         = r'X:\speciale\data\raw\%s\data\%s.db'%(data_handle,data_handle)
    events          = pd.read_csv(path).loc[:,'event_no'].reset_index(drop=True)
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