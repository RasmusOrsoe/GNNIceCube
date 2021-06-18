import pandas as pd
import numpy as np
import sqlite3
import torch
import random
import time 
from copy  import deepcopy
import multiprocessing
from multiprocessing import Pool
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from stellargraph import StellarGraph
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


def CreateStellarGraphs(setting):
    db_file = r'J:\speciale\data\raw\standard\dev_numu_train_l2_2020_01.db'
    scaler_E = torch.load(r'J:\speciale\data\minmax(1,10)scaler\target\scaler_E_special.pkl')
    scalers = pd.read_pickle(r'J:\speciale\data\raw\standard\transformers.pkl')
    scaler_mads = scalers['truth']['energy_log10']
    events = setting[0]
    path = setting[2]
    sca = pd.DataFrame()                                                             #
    seq = pd.DataFrame()
    feats = str('event_no,dom_x,dom_y,dom_z,dom_time,dom_charge')
    truths = str('event_no,energy_log10,time,position_x,position_y,position_z,direction_x,direction_y,direction_z,azimuth,zenith')                                                      #                                                      #
    with sqlite3.connect(db_file) as con:                                           #
        query = 'select %s from features WHERE event_no IN %s'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
        seq = seq.append(pd.read_sql(query, con))                       
        query = 'select %s from truth WHERE event_no IN %s'%(truths,str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
        sca = sca.append(pd.read_sql(query, con))                             #
        cursor = con.cursor()                                                       #
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    
    sca.loc[:,sca.columns[1]] = scaler_mads.inverse_transform(np.array(sca.loc[:,sca.columns[1]]).reshape(-1,1))
    gn = 0
    gc = 0
    now = time.time() 
    graphs = list()
    graphs_y = list()
    for event in range(0,len(sca['event_no'])): 
        index = seq['event_no'] == sca['event_no'][event] 
        x = pd.DataFrame(scaler.transform(seq.loc[index,['dom_x',
                           'dom_y',
                           'dom_z',
                           'dom_charge',
                           'dom_time']].sort_values('dom_time')))
        x.columns = ['dom_x','dom_y','dom_z','dom_time','dom_charge']
        x = x.reset_index(drop = True).sort_values('dom_time') 
        upper = x.index.values
        lower = np.roll(upper,-1)
        lower[len(lower)-1] = len(lower)-1
        x = np.array(x)
        y = sca.drop(columns = 'event_no').loc[event,:].values
        edge_index = pd.DataFrame({"source": upper, "target":lower})
        graphs.append(StellarGraph(x,edge_index))
        print('CREATING EVENT NR.:%s / %s' %(event,len(sca) ))
        gc = gc + 1
        if( gc == 100000 or event == len(sca)-1):
            print('Saving Graphs..')
            gc = 0
            gn = gn + 1
            label = multiprocessing.current_process().name[16:18] +'-'+ str(gn)
            s_path = path + '\\' + 'graph_event_only%s.pkl' %(label)
            torch.save(graphs,s_path)
            graphs = list()
            s_path_y = path + '\\' + 'event_only_truth%s.csv' %(label)
            sca.to_csv(s_path_y)
            graphs_y = list()
            
            
settings = list()
path = 'J:\\speciale\\data\\graphs\\standard\\sliced'
scaler = torch.load(r'J:\speciale\data\minmax(1,10)scaler\input\scaler_input(0,1).pkl')

events_array = pd.read_csv(r'J:\speciale\data\raw\standard\sliced\even_events.csv').loc[:,'event_no']#sca['event_no'][0:int(len(sca)/4)].values

random.shuffle(events_array)
event_list = np.array_split(events_array,1)
#settings = tuple([events_array,scaler,path])

for k in range(len(event_list)):
    settings.append(tuple([event_list[k],scaler,path]))
#CreateStellarGraphs(settings[0])
if __name__ == '__main__':
    p = Pool(processes = len(settings))
    start = time.time()
    async_result = p.map_async(CreateStellarGraphs, settings)
    p.close()
    p.join()
    print("Complete")
    end = time.time()
    print('total time (s)= ' + str(end-start))    