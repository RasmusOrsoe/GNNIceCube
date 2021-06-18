import time 
from torch_geometric.data import Data 
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from copy  import deepcopy
from multiprocessing import Pool
import multiprocessing
import sqlite3
import random
import torch
from collections import Counter
from numba import jit
#from numba import *
#import numba
#from numba.typed import List

#@jit(nopython = True) 
def event_only(seq,sca):
    x_result = []
    y_result = []
    edge_index_result = []
    for event in range(0,len(sca[:,0])): 
        index = np.where(seq[:,0] == sca[event,0])[0]
        x = seq[index,1:6]
        x[x[:,3].argsort()]
        upper = np.arange(0,len(x),1)
        lower = np.roll(upper,-1)
        lower[len(lower)-1] = len(lower)-1
        y = sca[event,1:12] 
        edge_index = [upper,lower]
        x_result.append(x)
        y_result.append(y)
        edge_index_result.append(edge_index)

    return x_result,y_result,edge_index_result            

def PullFromDB(db_file,events,feats,truths):
    sca = pd.DataFrame()                                                             #
    seq = pd.DataFrame()
    with sqlite3.connect(db_file) as con:                                           #
        query = 'select %s from features WHERE event_no IN %s'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
        seq = seq.append(pd.read_sql(query, con))                       
        query = 'select %s from truth WHERE event_no IN %s'%(truths,str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
        sca = sca.append(pd.read_sql(query, con))                             #
        cursor = con.cursor()                                                       #
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    sca.columns = ['event_no','energy_log10','time','position_x',
                   'position_y','position_z','direction_x',
                   'direction_y','direction_z','azimuth','zenith','pid']
    seq.columns = ['event_no','dom_x','dom_y','dom_z','dom_time','dom_charge']
    return sca,seq

def MakeGraph(setting):
    
    events,path,db_file,scalers, scaler,fullgrid,file_size,list_break, run_id, print_verbose = setting
    scalers = pd.read_pickle(scalers)
    scaler_mads = scalers['truth']['energy_log10']
    
    feats = str('event_no,x,y,z,time,charge_log10')
    truths = str('event_no,energy_log10,time,vertex_x,vertex_y,vertex_z,direction_x,direction_y,direction_z,azimuth,zenith,pid')                                                       #                                                      #
    gn = 0
    gc = 0
    print_count = 0
    graphs = list()
    array_print = pd.DataFrame(events)
    array_print.columns = ['event_no']
    if fullgrid == False :
        os.makedirs(path,exist_ok = True)
        print('EVENT ONLY' + ' ID: ' + multiprocessing.current_process().name[16:18])
        n_reps = int(np.ceil(len(events)/list_break))
        x_big = list()
        y_big = list()
        edge_index_big = list()
        for k in range(0,n_reps):
            if((k+1)*list_break > len(events)):
                up = len(events)
            else:
                up = (k+1)*list_break
                
            sca,seq = PullFromDB(db_file,(events[k*list_break:up]),feats,truths)
                                 
            sca.loc[:,sca.columns[1]] = scaler_mads.inverse_transform(np.array(sca.loc[:,sca.columns[1]]).reshape(-1,1))
            print(k)
            start = time.time()   
            x_mid,y_mid,edge_index_mid = event_only(np.array(seq),np.array(sca))
            print(time.time() - start)
            x_big.extend(x_mid)
            y_big.extend(y_mid)
            edge_index_big.extend(edge_index_mid)
        for event in range(0,len(events)): 
            x = scaler.transform(x_big[event])
            x = torch.tensor(np.array(x_big[event]).tolist(),dtype = torch.float) 
            y = torch.tensor(y_big[event].tolist(),dtype = torch.float) 
            edge_index = torch.tensor(edge_index_big[event], dtype = torch.long) 
            graphs.append( Data(x = x, edge_index = edge_index,y=  y.unsqueeze(0)))
            
            if(print_count  == print_verbose or event == len(events)-1):
                print('CREATING EVENT NR.:%s / %s' %(event,len(events) ))
                print_count = 0
                
            print_count +=1    
            gc = gc + 1
            if( gc == file_size or event == len(events)-1):
                print('Saving Graphs..')
                gc = 0
                label = str(run_id) + '-' + multiprocessing.current_process().name[16:18] +'-'+ str(gn)
                s_path = path + '\\' + 'graph_event_only%s.pkl' %(label)
                csv_path = path + '\\' + 'events_event_only%s.csv' %(label)
                torch.save(graphs,s_path)
                graphs = list()
                if((gn+1)*file_size > len(events)):
                    array_print['event_no'][gn*file_size:(len(events))].to_csv(csv_path)
                else:
                    array_print['event_no'][gn*file_size:(gn+1)*file_size].to_csv(csv_path)
                gn = gn + 1

### CONFIGURATION
events_array    = pd.read_csv(r'J:\speciale\data\raw\standard_reco\event_list\events.csv').loc[:,'event_no'][0:10000]
scaler          = torch.load(r'J:\speciale\data\minmax(1,10)scaler\input\scaler_input(0,1).pkl')
scalers         = r'J:\speciale\data\raw\standard_reco\transformers\transformers.pkl'
path            = 'J:\\speciale\\data\\graphs\\standard\\sliced\wadup'
db_file         = r'J:\speciale\data\raw\standard_reco\dev_numu_train_l5_retro_000.db'
fullgrid        = False
file_size       = 1000
list_break      = 1000
run_id          = 0
print_verbose   = 1000

### AUXILLIARY CONFIGURATION
n_workers = 1

settings = list()
np.random.shuffle(events_array)
event_list = np.array_split(events_array,n_workers)
for k in range(len(event_list)):
    settings.append(tuple([event_list[k],
                           path,
                           db_file,
                           scalers,
                           scaler,
                           fullgrid,
                           file_size,
                           list_break,
                           run_id,
                           print_verbose]))
start = time.time()
MakeGraph(settings[0])
print('total time (s)= ' + str(time.time()-start))
#if __name__ == '__main__':
#   p = Pool(processes = len(settings))
#   async_result = p.map_async(MakeGraph, settings)
#    p.close()
#    p.join()
#    print("Complete")
#    end = time.time()
#    print('total time (s)= ' + str(end-start))
