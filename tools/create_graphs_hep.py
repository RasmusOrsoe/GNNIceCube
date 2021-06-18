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

 
def MakeGraph(setting):
    db_file = '/groups/hep/pcs557/graph_factory/data/dev_numu_train_l5_retro_001.db'
    #scaler_E = torch.load(r'J:\speciale\data\minmax(1,10)scaler\target\scaler_E_special.pkl')
    scalers = pd.read_pickle(r'/groups/hep/pcs557/graph_factory/data/transformers.pkl')
    scaler_mads = scalers['truth']['energy_log10']
    events,node_size,path,file, SRT, edge_config, scaling, fullgrid, scaler,zero_pad = setting
    sca = pd.DataFrame()                                                             #
    seq = pd.DataFrame()
    feats = str('event_no,x,y,z,time,charge_log10')
    truths = str('event_no,energy_log10,time,vertex_x,vertex_y,vertex_z,direction_x,direction_y,direction_z,azimuth,zenith,pid')                                                       #                                                      #
    with sqlite3.connect(db_file) as con:                                           #
        query = 'select %s from features WHERE event_no IN %s'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
        seq = seq.append(pd.read_sql(query, con))                       
        query = 'select %s from truth WHERE event_no IN %s'%(truths,str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
        sca = sca.append(pd.read_sql(query, con))                             #
        cursor = con.cursor()                                                       #
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    h=0
    sca.columns = ['event_no','energy_log10','time','position_x',
                   'position_y','position_z','direction_x',
                   'direction_y','direction_z','azimuth','zenith','pid']
    seq.columns = ['event_no','dom_x','dom_y','dom_z','dom_time','dom_charge']
    #sca.loc[:,sca.columns[1:11]] = scaler_E.transform(sca.loc[:,sca.columns[1:11]])
    sca.loc[:,sca.columns[1]] = scaler_mads.inverse_transform(np.array(sca.loc[:,sca.columns[1]]).reshape(-1,1))
    gn = 0
    gc = 0
    now = time.time() 
    graphs = list() 
    if SRT == False and edge_config == 'Time' and fullgrid == False :
        print('EVENT ONLY')
        for event in range(0,len(sca['event_no'])): 
            index = np.where(seq['event_no'] == sca['event_no'][event])[0] 
            x = pd.DataFrame(scaler.transform(seq.loc[index,['dom_x',
                               'dom_y',
                               'dom_z',
                               'dom_charge',
                               'dom_time']].sort_values('dom_time')))
            x.columns = ['dom_x','dom_y','dom_z','dom_time','dom_charge']
            x = x.reset_index(drop = True).sort_values('dom_time')
            if(zero_pad == True):
                if(len(x) > node_size):
                    print('WARNING: SPECIFIED NODE SIZE TOO SMALL: %s'%len(x))
                    break
                diff = abs(len(x) - node_size)
                padding = pd.DataFrame(np.zeros((diff,len(x.columns))))
                padding.columns = x.columns
                x = x.append(padding)
                print(len(x))
            lol= 2
            x.loc[:,:] =  scaler.transform(x.loc[:,:])
            upper = x.index.values.tolist() 
            lower = np.roll(upper,-1).tolist() 
            lower[len(lower)-1] = len(lower)-1
            x = torch.tensor(np.array(x).tolist(),dtype = torch.float) 
            y = torch.tensor(sca.drop(columns = 'event_no').loc[event,:].values,dtype = torch.float) 
            edge_index = torch.tensor([upper, lower], dtype = torch.long) 
            graphs.append( Data(x = x, edge_index = edge_index,y=  y.unsqueeze(0)))
            print('CREATING EVENT NR.:%s / %s' %(event,len(sca) ))
            gc = gc + 1
            if( gc == 100000 or event == len(sca)-1):
                print('Saving Graphs..')
                gc = 0
                gn = gn + 1
                label = multiprocessing.current_process().name[16:18] +'-'+ str(gn)
                s_path = path + '\\' + 'graph_event_only%s.pkl' %(label)
                csv_path = path + '\\' + 'events_event_only%s.csv' %(label)
                torch.save(graphs,s_path)
                graphs = list()
                sca['event_no'].to_csv(csv_path)
            
    if SRT == False and edge_config == 'Time' and fullgrid == True :
        print('FULLD GRID CONFIGURATION')
        gn = 0
        gc = 0
        graphs = list()
        bare_dir = 'J:\\speciale\\data\\raw\\standard\\sliced'
        bare_graph = pd.read_csv(bare_dir+'\\bare_graph.csv').loc[:,['dom_x','dom_y','dom_z','dom_charge','dom_time']]
        
        grid = list()
        for k in range(0,len(bare_graph)):
            grid.append(tuple(bare_graph.loc[k,bare_graph.columns[1:4]]))
        grid = pd.Series(grid)
        
        for event in range(0,len(sca)):
            empty_graph = deepcopy(bare_graph)    
            index = seq['event_no'] == sca['event_no'][event]
            x = seq.loc[index,['dom_x',
                               'dom_y',
                               'dom_z',
                               'dom_charge',
                               'dom_time']].sort_values('dom_time').reset_index(drop = True)
            for node in range(0,len(x)):
                index = grid == tuple(x.loc[node,['dom_x','dom_y','dom_z']])
                if sum(index) == 1:
                    empty_graph.loc[node,['dom_charge','dom_time']] =  x.loc[node,['dom_charge','dom_time']]
                if sum(index) > 1:
                    empty_graph = empty_graph.append(x.loc[node,['dom_x','dom_y','dom_z','dom_charge','dom_time']])
            empty_graph = empty_graph.reset_index(drop=True)
            empty_graph = empty_graph.sort_values('dom_time')
            
            upper = empty_graph.index.values.tolist() 
            lower = np.roll(upper,-1) 
            lower[len(lower)-1] = len(lower)-1
            
            x = torch.tensor(np.array(empty_graph).tolist(),dtype = torch.float)
            y = torch.tensor(sca.drop(columns = 'event_no').loc[event,:].values,dtype = torch.float) 
            edge_index = torch.tensor([upper, lower], dtype = torch.long)
            graphs.append( Data(x = x, edge_index = edge_index,y =  y.unsqueeze(0)))
            print(event)
            gc = gc + 1
            if( gc == 10000 or event == len(sca)):
                gc = 0
                gn = gn + 1
                s_path = path + '\\' + 'graph_test' +'full_grid%s.pkl' %(gn)
                torch.save(graphs,s_path)
                graphs = list()
            
                
                
            
        
    
        
    return print('GRAPHS CREATED: %s || %s '%(s_path,abs(time.time() - now))); 


path = '/groups/hep/pcs557/graph_factory/graphs'

settings = list()
events_array = pd.read_csv('/groups/hep/pcs557/graph_factory/data/events.csv').loc[:,'event_no']
    
scaler = 0 #torch.load(r'J:\speciale\data\minmax(1,10)scaler\input\scaler_input(0,1).pkl')
#seq_events = pd.DataFrame()
#db_file = r'J:\speciale\data\raw\standard_reco\dev_numu_train_l5_retro_000.db'
#with sqlite3.connect(db_file) as con:                                           #                      
#    query = 'select event_no from features where event_no in %s'%str(tuple(events_array))                                              # THESE ARE THEN WRITTEN TO DRIVE
#    seq_events = seq_events.append(pd.read_sql(query, con))                             #
#    cursor = con.cursor()                                                       #
#    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

max_nodes = 0 # max(Counter(seq_events['event_no']).values())    
    
#sca['event_no'][0:int(len(sca)/4)].values
random.shuffle(events_array)
event_list = np.array_split(events_array,4)
for k in range(len(event_list)):
    settings.append(tuple([event_list[k],max_nodes,path,'scaled',False,'Time','Standard', False,scaler,False]))
MakeGraph(settings[0])

#if __name__ == '__main__':
#    p = Pool(processes = len(settings))
#    start = time.time()
#    async_result = p.map_async(MakeGraph, settings)
#    p.close()
#    p.join()
#    print("Complete")
#    end = time.time()
#    print('total time (s)= ' + str(end-start))

