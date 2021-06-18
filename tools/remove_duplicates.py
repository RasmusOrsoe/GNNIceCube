import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os
import time
from multiprocessing import Pool
import multiprocessing
pd.set_option('mode.chained_assignment',None)

start_time = time.time()
def MakeDuplicateBool(settings):
    bad_events, mc_db, worker_id = settings
    stat = []
    removed_list = []
    event_nos = []
    row  = []
    for j in range(0,len(bad_events)):
        with sqlite3.connect(mc_db) as con:
            query = 'select event_no, string, dom,row from features where event_no = %s and SRTInIcePulses = 1'%(bad_events[j])
            data = pd.read_sql(query,con)
            
        dom_id_list = pd.DataFrame(deepcopy(data['dom']))
        dom_id_list.columns = ['duplicate_screen']
        event_nos.extend(data['event_no'])
        row.extend(data['row'])
        #unique_id = pd.unique(dom_id_list['duplicate_screen'])
        pairs = pd.concat([data['string'],data['dom']], axis = 1)
        pairs = list(pairs.values)
        lol = list(map(tuple,pairs))
        unique_pairs = pd.unique(lol)
        

        pairs = list(map(list,pairs))
        pairs = np.array(list(pairs))
        #stat = []
        count = 0
        for i in range(0,len(unique_pairs)):
            #print(pairs == unique_pairs[i])
            index  = (pairs == unique_pairs[i]).sum(1) == 2
            if sum(index) > 3:
                #first_occ = dom_id_list['duplicate_screen'][index].index[0]
                #dom_id_list['duplicate_screen'][index] = 0
                #dom_id_list['duplicate_screen'][first_occ] = 1
                count = count +sum(index)
            #else:
                #dom_id_list[index] = 1
        
        stat.append([count,len(pairs)])
        
        #removed_list.extend(dom_id_list['duplicate_screen'])
        print('%s / %s'%(j,len(bad_events)))
                
                
    #results = pd.concat([pd.DataFrame(row),pd.DataFrame(event_nos), pd.DataFrame(removed_list)],axis = 1)
    #results.columns = ['row','event_no','R-filter']
    save_dir = r'X:\speciale\data\export\Upgrade Cleaning\statcount_bad'
    os.makedirs(save_dir, exist_ok = True)
    #results.to_csv(save_dir + '\\stat%s.csv'%worker_id) 
    pd.DataFrame(stat).to_csv(save_dir + '\\stat%s.csv'%worker_id)



if __name__ == '__main__':
    event_path = r'X:\speciale\data\export\Upgrade Cleaning\duplicates\bad\events.csv'#'/groups/hep/pcs557/speciale/upgrade_clean/selections/martin_cut/martin_selection.csv'
    events = pd.read_csv(event_path)['event_no'].reset_index(drop = True)[0:1000000]
    mc_db = r'X:\speciale\data\raw\dev_upgrade_train_step4_001\data\dev_upgrade_train_step4_001.db'

    n_workers = 5
    #scalers = pd.read_pickle('/groups/hep/pcs557/speciale/data/raw/dev_upgrade_train_step4_001/meta/transformers.pkl')

    with sqlite3.connect(mc_db) as con:
        query = 'select event_no from truth where event_no in %s'%(str(tuple(events)))
        data_main = pd.read_sql(query,con)


    settings = []
    event_list = np.array_split(data_main['event_no'].reset_index(drop = True),n_workers)
    for i in range(0,n_workers):
        settings.append([event_list[i].reset_index(drop = True), mc_db, i])
    #MakeDuplicateBool(settings[0])
    p = Pool(processes = n_workers)
    async_res = p.map_async(MakeDuplicateBool,settings)
    p.close()
    p.join()
    print('Complete')
    print('Total Time: %s (s)'%(time.time() - start_time))
            
        

    

#counter = counter + 1
    

        
    


