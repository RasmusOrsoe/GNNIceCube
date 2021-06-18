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
#from numba import jit
#from numba import *
#import numba
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#@jit(nopython = True) 
def event_only(seq,sca,hw,lw,mode,model_low,model_high,model_nw):
    if mode == 'mc':
        x_result = []
        y_result = []
        edge_index_result = []
        hw_result      = []
        lw_result      = []
        model_low_res  = []
        model_high_res = []
        model_nw_res   = []
        for event in range(0,len(sca[:,0])): 
            index = np.where(seq[:,0] == sca[event,0])[0]
            x = seq[index,1:]
            x[x[:,3].argsort()]
            upper = np.arange(0,len(x),1)
            #print(len(x))
            lower = np.roll(upper,-1)
            lower[len(lower)-1] = len(lower)-1
            y = sca[event,1:] 
            edge_index = [upper,lower]
            if(len(model_low) != 0):
                index_lw = np.where(model_low[:,0] == sca[event,0])[0]
                model_lw = model_low[index_lw,1:]
                index_high = np.where(model_high[:,0] == sca[event,0])[0]
                model_hg   = model_high[index_high,1:]
                index_nw = np.where(model_nw[:,0] == sca[event,0])[0]
                model_nws = model_nw[index_nw,1:]
                
                model_low_res.append(model_lw)
                model_high_res.append(model_hg)
                model_nw_res.append(model_nws)
            
            x_result.append(x)
            y_result.append(y)
            edge_index_result.append(edge_index)
            hw_result.append(hw[event])
            lw_result.append(lw[event])
    
    if mode == 'data':
        x_result = []
        y_result = []
        edge_index_result = []
        hw_result = []
        lw_result = []
        model_low_res  = []
        model_high_res = []
        model_nw_res   = []
        for event in range(0,len(sca)): 
            index = np.where(seq[:,0] == sca[event])[0]
            x = seq[index,1:]
            x[x[:,3].argsort()]
            upper = np.arange(0,len(x),1)
            lower = np.roll(upper,-1)
            lower[len(lower)-1] = len(lower)-1
            y = np.array([0]) 
            edge_index = [upper,lower]
            
            if(len(model_low) != 0):
                index_lw = np.where(model_low[:,0] == sca[event,0])[0]
                model_lw = model_low[index_lw,1:]
                index_high = np.where(model_high[:,0] == sca[event,0])[0]
                model_hg   = model_high[index_high,1:]
                index_nw = np.where(model_nw[:,0] == sca[event,0])[0]
                model_nws = model_nw[index_nw,1:]
                
                model_low_res.append(model_lw)
                model_high_res.append(model_hg)
                model_nw_res.append(model_nws)
            
            
            x_result.append(x)
            y_result.append(y)
            edge_index_result.append(edge_index)
            hw_result.append(1)
            lw_result.append(1)
            
            
    return x_result,y_result,edge_index_result,[sca],hw_result,lw_result, model_low_res,model_high_res,model_nw_res  
          
def Generate_Event_Array(path,db_file, mode):
    check = os.path.isfile(path +  '\events\events.csv')
    if(check == False):
        print('NO EVENT ARRAY FOUND. GENERATING...')
        os.makedirs(path+'\events',exist_ok = True)
        if mode == 'mc':
            with sqlite3.connect(db_file) as con:                                           #
                query = 'select event_no from truth' 
                truth = pd.read_sql(query,con)
        if mode == 'data':
            with sqlite3.connect(db_file) as con:                                           #
                query = 'select event_no from features' 
                event_nos = pd.read_sql(query,con)
            truth = pd.DataFrame(event_nos['event_no'].unique())
            truth.columns = ['event_no']
        truth.to_csv(path + '\events\events.csv')
        print('EVENT ARRAY CREATED: \n Events: %s \n Path: %s'%(len(truth),path+'\events\events.csv'))
    else:
        print('EVENT ARRAY FOUND AT: \n %s' %(path+'\events\events.csv'))
    return path+'\events\events.csv'

def PullFromDB(db_file,events,feats,truths,upgrade,SRT,classification,mode, ensemble, handle):
    #print('in pulldb')
    sca = pd.DataFrame()                                                             #
    seq = pd.DataFrame()
    model_low = []
    model_high = []
    model_nw = []
    if mode == 'mc':
        #### THIS IS MONTE-CARLO DATA
        if ensemble == True:
            ensemble_db = r'X:\speciale\data\graphs' + '\\' + handle + '\\' + 'predictions\\ensemble_predictions.db' 
            with sqlite3.connect(ensemble_db) as con:
                query = 'select * from dynedgev3_multi_low WHERE event_no IN %s '%(str(tuple(events))) 
                model_low = pd.read_sql(query, con)
                query = 'select * from dynedgev3_multi_high WHERE event_no IN %s '%(str(tuple(events))) 
                model_high = pd.read_sql(query, con)
                query = 'select * from dynedgev3_multi_nw WHERE event_no IN %s '%(str(tuple(events))) 
                model_nw = pd.read_sql(query, con)
        
        if SRT == False:
            #print('grabbin data')
            with sqlite3.connect(db_file) as con:                                           #
                query = 'select %s from features WHERE event_no IN %s'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
                seq = seq.append(pd.read_sql(query, con))                       
                query = 'select %s from truth WHERE event_no IN %s'%(truths,str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
                sca = sca.append(pd.read_sql(query, con))                             #
                cursor = con.cursor()                                                       #
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            #print('data grabbed')
        if SRT == True and upgrade == False:
            with sqlite3.connect(db_file) as con:                                           #
                query = 'select %s from features WHERE event_no IN %s and SRTInIcePulses = 1'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
                seq = seq.append(pd.read_sql(query, con))                       
                query = 'select %s from truth WHERE event_no IN %s'%(truths,str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
                sca = sca.append(pd.read_sql(query, con))                             #
                cursor = con.cursor()                                                       #
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        if upgrade == True and SRT == True:
            with sqlite3.connect(db_file) as con:                                           #
                query = 'select %s from features WHERE event_no IN %s and SRTInIcePulses = 1'%(feats,str(tuple(events)))# and lc = 1'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
                seq = seq.append(pd.read_sql(query, con))                       
                query = 'select %s from truth WHERE event_no IN %s'%(truths,str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
                sca = sca.append(pd.read_sql(query, con))                             #
                cursor = con.cursor()                                                       #
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            
        if(classification == False and upgrade == False):    
            #sca.columns = ['event_no','energy_log10','time','position_x',
            #               'position_y','position_z','direction_x',
            #               'direction_y','direction_z','azimuth','zenith','pid']
            sca.columns = ['event_no','energy_log10','position_x',
                           'position_y','position_z','azimuth','zenith','pid']
        if( classification == True and upgrade == False):
            #print('in class')
            #sca.columns = ['event_no','energy_log10','time','position_x',
            #               'position_y','position_z','direction_x',
            #               'direction_y','direction_z','azimuth','zenith','pid','stopped_muon']
            sca.columns = ['event_no','energy_log10','position_x',
                           'position_y','position_z',
                           'azimuth','zenith','pid', 'interaction_type']
            
            #holder = deepcopy(sca['stopped_muon'])
            #holder[holder.isnull()] = 0
            #sca['stopped_muon'] = holder.astype(int)
            #### ID TABLE ###############
            # muon        : 0  #  0 # 0 #
            # stopped_muon: 1  #  1 # 0 #
            # v_e         : 2  #  2 # 1 #
            # v_u         : 3  #  2 # 1 #
            # v_t         : 4  #  2 # 1 #
            #############################
            muon_index         = abs(deepcopy(sca['pid']))     == 13
            #muon_stopped_index = deepcopy(sca['stopped_muon']) == 1
            #corsika_index      = abs(deepcopy(sca['pid']))     > 20
            noise_index        = abs(deepcopy(sca['pid']))     == 1
            v_e_index          = abs(deepcopy(sca['pid']))     == 12
            v_u_index          = abs(deepcopy(sca['pid']))     == 14
            v_t_index          = abs(deepcopy(sca['pid']))     == 16
            track_index        = (abs(sca['pid']) == 14) & (sca['interaction_type'] == 1)
            
            pid_holder = deepcopy(sca['pid'])
            #pid_holder[muon_index]         = 0
            #pid_holder[noise_index]        = 0
            #pid_holder[muon_stopped_index] = 0#1
            #pid_holder[corsika_index]      = 0
            #pid_holder[v_e_index]          = 1#2
            #pid_holder[v_u_index]          = 1#2 #3
            #pid_holder[v_t_index]          = 1#2 #4
            pid_holder[track_index] = 1
            pid_holder[~track_index] = 0

            sca['pid'] = pid_holder
            sca = sca.drop(columns = ['interaction_type'])
            
        if(upgrade == True):
            seq.columns = feats.split(',')
        else:
            seq.columns = feats.split(',')
            
        
           
        
            
        return sca,seq, model_low,model_high,model_nw
    if mode == 'data':
        if ensemble == True:
            ensemble_db = r'X:\speciale\data\graphs' + '\\' + handle + '\\' + 'predictions\\ensemble_predictions.db' 
            with sqlite3.connect(ensemble_db) as con:
                query = 'select * from dynedgev3_multi_low WHERE event_no IN %s '%(str(tuple(events))) 
                model_low = pd.read_sql(query, con)
                query = 'select * from dynedgev3_multi_high WHERE event_no IN %s '%(str(tuple(events))) 
                model_high = pd.read_sql(query, con)
                query = 'select * from dynedgev3_multi_nw WHERE event_no IN %s '%(str(tuple(events))) 
                model_nw = pd.read_sql(query, con)
        ### THIS IS REAL MEASUREMENTS FROM ICECUBE
        if SRT == False:
            with sqlite3.connect(db_file) as con:                                           #
                query = 'select %s from features WHERE event_no IN %s'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
                seq = seq.append(pd.read_sql(query, con))                       
        if SRT == True:
            with sqlite3.connect(db_file) as con:                                           #
                query = 'select %s from features WHERE event_no IN %s and SRTInIcePulses = 1'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
                seq = seq.append(pd.read_sql(query, con))
        if(upgrade == True):
            seq.columns = feats.split(',')
        else:
            seq.columns = feats.split(',')
        sca = seq['event_no'].unique()                               
        return sca,seq, model_low,model_high,model_nw        
def MakeGraph(setting):
    event_info,path,db_file,scalers, scaler,fullgrid,file_size,list_break, run_id, print_verbose, GRU,SRT, upgrade,classification,mode, data_handle, ensemble,graph_handle = setting
    print('SRT: %s'%SRT)
    handle = data_handle + '\\' + graph_handle
    events = event_info[:,0]
    low_weights = event_info[:,1]
    high_weights = event_info[:,2]
    scalers = pd.read_pickle(scalers)
    input_scalers = []
    if mode == 'mc':    
        scaler_mads = scalers['truth']['energy_log10']
    #events = events.reset_index(drop = True)
    
    feats = str('event_no,dom_x,dom_y,dom_z,dom_time,charge_log10,rqe,width,pmt_area') #str('event_no,x,y,z,time,charge_log10')
    truths = str('event_no,energy_log10,position_x,position_y,position_z,azimuth,zenith,pid')  #str('event_no,energy_log10,time,vertex_x,vertex_y,vertex_z,direction_x,direction_y,direction_z,azimuth,zenith,pid')                                                       #                                                      #
    if classification ==  True:
        feats = str('event_no,dom_x,dom_y,dom_z,dom_time,charge_log10,rqe,width,pmt_area')
        #truths = str('event_no,energy_log10,time,position_x,position_y,position_z,direction_x,direction_y,direction_z,azimuth,zenith,pid,stopped_muon')
        truths = str('event_no,energy_log10,position_x,position_y,position_z,azimuth,zenith,pid, interaction_type')
    
    if( upgrade == True):
        #feats = str('event_no,dom_x,dom_y,dom_z,time,charge_log10,string,pmt,lc,pmt_type,pmt_area,dom,pmt_x,pmt_y,pmt_z')
        feats = str('event_no,dom_x,dom_y,dom_z,time,charge_log10')
        truths = str('event_no,energy_log10,position_x,position_y,position_z,azimuth,zenith,pid')# str('event_no,energy_log10,time,position_x,position_y,position_z,direction_x,direction_y,direction_z,azimuth,zenith,pid')
    if (mode == 'data'):
        feats = str('event_no,dom_x,dom_y,dom_z,dom_time,charge_log10,rqe,width,pmt_area')#feats = str('event_no,dom_x,dom_y,dom_z,dom_time,charge_log10')    
        
    features = feats.split(',')
    gn = 0
    gc = 0
    print_count = 0
    graphs = list()
    array_print = pd.DataFrame(events)
    array_print.columns = ['event_no']
    if fullgrid == False and GRU == False :
        os.makedirs(path,exist_ok = True)
        print('EVENT ONLY' + ' ID: ' + multiprocessing.current_process().name[16:18])
        n_reps = int(np.ceil(len(events)/list_break))
        x_big = list()
        y_big = list()
        edge_index_big = list()
        event_no_big = list()
        lw_big = list()
        hw_big = list()
        
        model_low_big = list()
        model_high_big = list()
        model_nw_big   = list()
        
        k_count = 0
        for k in range(0,n_reps):
            if((k+1)*list_break > len(events)):
                up = len(events)
            else:
                up = (k+1)*list_break
                
            sca,seq,model_low,model_high,model_nw = PullFromDB(db_file,(events[k*list_break:up]),feats,truths,upgrade,SRT,classification,mode, ensemble, handle)
            if len(sca) == 0:
                print('shit')
            lw = low_weights[k*list_break:up]
            hw = high_weights[k*list_break:up]
            
            ## SCALING
            #scalers = pd.read_pickle(r'X:\speciale\data\raw\%s\meta\transformers.pkl'%data_handle)['truth']
            #sca.loc[:,'zenith'] = scalers['zenith'].inverse_transform(np.array(sca.loc[:,'zenith']).reshape(-1,1))                     
            #sca.loc[:,'azimuth'] = scalers['azimuth'].inverse_transform(np.array(sca.loc[:,'azimuth']).reshape(-1,1))
            #sca.loc[:,'energy_log10'] = scalers['energy_log10'].inverse_transform(np.array(sca.loc[:,'energy_log10']).reshape(-1,1))  
            #if( mode == 'mc'):
            #    sca.loc[:,sca.columns[1]] = scaler_mads.inverse_transform(np.array(sca.loc[:,sca.columns[1]]).reshape(-1,1))
            
            ##
            #if(k_count == int(print_verbose/20)):
            lol = 0
            print('WORKER %s : %s / %s' %(multiprocessing.current_process().name[16:18],k,n_reps))  
            x_mid,y_mid,edge_index_mid,event_no_mid,hw_mid,lw_mid,model_low_mid, model_high_mid, model_nw_mid = event_only(np.array(seq),
                                                                                                                           np.array(sca),
                                                                                                                           hw,
                                                                                                                           lw,
                                                                                                                           mode,
                                                                                                                           np.array(model_low),
                                                                                                                           np.array(model_high),
                                                                                                                           np.array(model_nw))
            x_big.extend(x_mid)
            y_big.extend(y_mid)
            edge_index_big.extend(edge_index_mid)
            event_no_big.extend(np.array(event_no_mid).flatten())
            lw_big.extend(lw_mid)
            hw_big.extend(hw_mid)
            
            if ensemble == True:
                model_low_big.extend(model_low_mid)
                model_high_big.extend(model_high_mid)
                model_nw_big.extend(model_nw_mid)
                
            k_count += 1
        for event in range(0,len(events)):
            x = x_big[event]
            x = torch.tensor(np.array(x).tolist(),dtype = torch.float) 
            y = torch.tensor(y_big[event].tolist(),dtype = torch.float) 
            edge_index = torch.tensor(edge_index_big[event], dtype = torch.long)
            event_no = torch.tensor(events[event],dtype = torch.int64)
            lw = torch.tensor(lw_big[event],dtype = torch.float)
            hw = torch.tensor(hw_big[event],dtype = torch.float)
            
            if ensemble == True:
                mdl_low = torch.tensor(model_low_big[event],dtype = torch.float)
                mdl_high = torch.tensor(model_high_big[event],dtype = torch.float)
                mdl_nw = torch.tensor(model_nw_big[event],dtype = torch.float)
            
            if ensemble  == False:
                if mode == 'mc':
                    mc = 1
                else:
                    mc = 0
                graphs.append( Data(x = x,
                                    edge_index = edge_index,
                                    y=  y.unsqueeze(0),
                                    event_no = torch.tensor(event_no,dtype = torch.int64),
                                    hw = hw,
                                    lw = lw,
                                    mc = mc))
            if ensemble == True:
                if(len(model_low_big) == 0):
                    graphs.append( Data(x = x,
                                    edge_index = edge_index,
                                    y=  y.unsqueeze(0),
                                    event_no = torch.tensor(event_no,dtype = torch.int64),
                                    hw = hw,
                                    lw = lw))
                if len(model_low_big) != 0:
                    graphs.append( Data(x = x,
                                    edge_index = edge_index,
                                    y=  y.unsqueeze(0),
                                    event_no = torch.tensor(event_no,dtype = torch.int64),
                                    hw = hw,
                                    lw = lw,
                                    model_low  = mdl_low,
                                    model_high = mdl_high,
                                    model_nw   = mdl_nw))
                
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
                
                
    if fullgrid == False and GRU == True :
        os.makedirs(path,exist_ok = True)
        print('EVENT ONLY' + ' ID: ' + multiprocessing.current_process().name[16:18])
        n_reps = int(np.ceil(len(events)/list_break))
        x_big = list()
        y_big = list()
        edge_index_big = list()
        event_no_big = list()
        lw_big = list()
        hw_big = list()
        k_count = 0
        for k in range(0,n_reps):
            if((k+1)*list_break > len(events)):
                up = len(events)
            else:
                up = (k+1)*list_break
                
            sca,seq = PullFromDB(db_file,(events[k*list_break:up]),feats,truths,upgrade)
            lw = low_weights[k*list_break:up]
            hw = high_weights[k*list_break:up]
            #for input_scaler in range(0,len(input_scalers)):
            #    seq.loc[:,seq.columns[input_scaler+1]] = input_scalers[input_scaler].inverse_transform(np.array(seq.loc[:,seq.columns[input_scaler+1]]).reshape(-1,1))                   
            sca.loc[:,sca.columns[1]] = scaler_mads.inverse_transform(np.array(sca.loc[:,sca.columns[1]]).reshape(-1,1))
            if(k_count == int(print_verbose/20)):
                print('WORKER %s : %s / %s' %(multiprocessing.current_process().name[16:18],k,n_reps))  
            x_mid,y_mid,edge_index_mid,event_no_mid,hw_mid,lw_mid = event_only(np.array(seq),np.array(sca),hw,lw)
            x_big.extend(x_mid)
            y_big.extend(y_mid)
            edge_index_big.extend(edge_index_mid)
            event_no_big.extend(np.array(event_no_mid).flatten())
            lw_big.extend(lw_mid)
            hw_big.extend(hw_mid)
            
            k_count += 1
        for event in range(0,len(events)):
            if(len(x_big[event]) <= 400):
                x = x_big[event]
                diff = abs(len(x_big[event]) - 400)
                padding = np.zeros((diff,x.shape[1]))
                x = np.concatenate((np.array(x),padding))
                x = torch.tensor(x.tolist(),dtype = torch.float) 
                y = torch.tensor(y_big[event].tolist(),dtype = torch.float) 
                edge_index = torch.tensor(edge_index_big[event], dtype = torch.long)
                event_no = torch.tensor(events[event],dtype = torch.int64)
                lw = torch.tensor(lw_big[event],dtype = torch.float)
                hw = torch.tensor(hw_big[event],dtype = torch.float)
                graphs.append( Data(x = x,
                                edge_index = edge_index,
                                y=  y.unsqueeze(0),
                                event_no = event_no_big[event],
                                hw = hw,
                                lw = hw))
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
            
    # if fullgrid == True :
    #     print('FULLD GRID CONFIGURATION')
    #     gn = 0
    #     gc = 0
    #     graphs = list()
    #     bare_dir = 'J:\\speciale\\data\\raw\\standard\\sliced'
    #     bare_graph = pd.read_csv(bare_dir+'\\bare_graph.csv').loc[:,['dom_x','dom_y','dom_z','dom_charge','dom_time']]
        
    #     grid = list()
    #     for k in range(0,len(bare_graph)):
    #         grid.append(tuple(bare_graph.loc[k,bare_graph.columns[1:4]]))
    #     grid = pd.Series(grid)
        
    #     for event in range(0,len(sca)):
    #         empty_graph = deepcopy(bare_graph)    
    #         index = seq['event_no'] == sca['event_no'][event]
    #         x = seq.loc[index,['dom_x',
    #                            'dom_y',
    #                            'dom_z',
    #                            'dom_charge',
    #                            'dom_time']].sort_values('dom_time').reset_index(drop = True)
    #         for node in range(0,len(x)):
    #             index = grid == tuple(x.loc[node,['dom_x','dom_y','dom_z']])
    #             if sum(index) == 1:
    #                 empty_graph.loc[node,['dom_charge','dom_time']] =  x.loc[node,['dom_charge','dom_time']]
    #             if sum(index) > 1:
    #                 empty_graph = empty_graph.append(x.loc[node,['dom_x','dom_y','dom_z','dom_charge','dom_time']])
    #         empty_graph = empty_graph.reset_index(drop=True)
    #         empty_graph = empty_graph.sort_values('dom_time')
            
    #         upper = empty_graph.index.values.tolist() 
    #         lower = np.roll(upper,-1) 
    #         lower[len(lower)-1] = len(lower)-1
            
    #         x = torch.tensor(np.array(empty_graph).tolist(),dtype = torch.float)
    #         y = torch.tensor(sca.drop(columns = 'event_no').loc[event,:].values,dtype = torch.float) 
    #         edge_index = torch.tensor([upper, lower], dtype = torch.long)
    #         graphs.append( Data(x = x, edge_index = edge_index,y =  y.unsqueeze(0)))
    #         print(event)
    #         gc = gc + 1
    #         if( gc == 10000 or event == len(sca)):
    #             gc = 0
    #             gn = gn + 1
    #             s_path = path + '\\' + 'graph_test' +'full_grid%s.pkl' %(gn)
    #             torch.save(graphs,s_path)
    #             graphs = list()
            
   #return print('GRAPHS CREATED: %s || %s '%(s_path,abs(time.time() - now))); 
def FitScaler(db_file,events,path,upgrade):
    check = os.path.isfile(path + '\\input_scaler\\input_scaler%s.pkl'%int(len(events)/1000))
    if(check == False):
        print('Scaler not found. Fitting scaler..')                                                            #
        seq = pd.DataFrame()
        feats = str('x,y,z,time,charge_log10')
        if(upgrade == True):
            feats = str('dom_x,dom_y,dom_z,dom_time,dom_charge')
            #feats = str('dom_x,dom_y,dom_z,dom_time,dom_charge,dom_string,dom_pmt,dom_om,dom_lc,dom_atwd,dom_fadc')
        with sqlite3.connect(db_file) as con:                                           #
            query = 'select %s from features WHERE event_no IN %s'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
            seq = seq.append(pd.read_sql(query, con))                       
        #if(upgrade == True):
        #    seq.columns = ['event_no','dom_x','dom_y','dom_z','dom_time','dom_charge',
        #                   'dom_string','dom_pmt','dom_om','dom_lc','dom_atwd','dom_fadc']
        #else:
        #    seq.columns = ['event_no','dom_x','dom_y','dom_z','dom_time','dom_charge']
        scaler = MinMaxScaler(feature_range=(0,1)).fit(seq)
        del seq
        print('scaler fitted')
        os.makedirs(path + '\\input_scaler',exist_ok = True)
        torch.save(scaler,path + '\\input_scaler\\input_scaler%s.pkl'%int(len(events)/1000))
        print('Scaler saved to: \n \
              %s'%(path + '\\input_scaler\\input_scaler%s.pkl'%int(len(events)/1000)))
    else:
        scaler = torch.load(path + '\\input_scaler\\input_scaler%s.pkl'%int(len(events)/1000))          
    return scaler

def CreateScalers(db_file,graph_handle,db_handle,features):   
    with sqlite3.connect(db_file) as con2:
        query = 'SELECT Count(*) FROM features'
        n_rows = np.array(pd.read_sql(query, con2))
    indicies = np.arange(1,n_rows)
    np.random.shuffle(indicies)
    rows = indicies[0:10000000]
    for feature in features:
        check = os.path.isfile(r'X:\speciale\data\graphs\%s\input_scalers\%s_scaler.pkl'%(db_handle,feature))
        print('Checking for %s scaler'%feature)
        if(check == True):
            print('Found %s scaler'%feature)
        if(check == False):
            print('%s scaler not found.. Generating..'%feature)
            seq = pd.DataFrame()
            with sqlite3.connect(db_file) as con:                                           #
                query = 'select %s from features where rowid IN %s'%(feature,str(tuple(rows)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
                seq = seq.append(pd.read_sql(query, con))                       
            scaler = MinMaxScaler(feature_range = (0,1)).fit(seq)
            os.makedirs('X:\speciale\data\graphs\%s\input_scalers'%(db_handle),exist_ok = True)
            torch.save(scaler,'X:\speciale\data\graphs\%s\input_scalers\%s_scaler.pkl'%(db_handle,feature))
    return

def CreateWeights(n_bins,db_file,scalers,events,data_handle,graph_handle,x_low,type,mode):
    print('Creating Weights: \n \
          profile : %s \n \
          events  : %s'%(type,len(events)))
    if mode == 'mc':
        sca = pd.DataFrame()
        scalers = pd.read_pickle(scalers)
        scaler_mads = scalers['truth']['zenith']                                                             
        with sqlite3.connect(db_file) as con:                                                                 
            query = 'select zenith from truth'                                              
            sca = sca.append(pd.read_sql(query, con))
        sca = scaler_mads.inverse_transform(np.array(sca).reshape(-1,1))
        #fig = plt.figure()
        #N, bins, patches = plt.hist(sca, bins = 50)
        #plt.close()
        #index  = np.digitize(sca, bins=bins)
        #print(index.shape)
        #unique_bins = pd.unique(pd.DataFrame(index[:,0]).loc[:,0])
        #unique_bins.sort()
        #bin_count = []
        #for unique_bin in unique_bins:
        #    bin_count.append(sum(index == unique_bin))
        weights  = []
        print('in')
        for i in range(0,len(sca)):
            weights.append(float(weight(sca[i],x_low,type,1, 1)))
        diff = 1 - np.mean(weights)
        weights = np.array(weights) + diff
        print(np.mean(weights))
        print(sum(weights))
        print('Weights are done!')
        os.makedirs(r'X:\speciale\data\graphs\%s\%s\weights'%(data_handle,graph_handle),exist_ok = True)
        pd.DataFrame(weights).to_csv(r'X:\speciale\data\graphs\%s\%s\weights\weights_%s.csv'%(data_handle,graph_handle,type),index = True)              
    
    if mode == 'data':
        weights = np.repeat(1,len(events))
        print(np.mean(weights))
        print(sum(weights))
        print('Weights are done!')
        os.makedirs(r'X:\speciale\data\graphs\%s\%s\weights'%(data_handle,graph_handle),exist_ok = True)
        pd.DataFrame(weights).to_csv(r'X:\speciale\data\graphs\%s\%s\weights\weights_%s.csv'%(data_handle,graph_handle,type),index = True)              
    return weights

def weight(x,x_threshold,type, bin_count, bins):
    if type == 'low':
        if x<x_threshold:
            return 1
    
        if x>= x_threshold:
            return 1*(1/(1+ x-x_threshold))
    if type == 'high':
        if(x> x_threshold):
            return 1
        if(x< x_threshold):
            return 1*(1/(1+x_threshold-x))
    if type == 'inverse_count':
        bin_idx = np.digitize(x, bins=bins)
        #print(bin_idx)
        #print(bin_count)
        return 1/(sum(bin_count[bin_idx[0]-1]))
  
    
if __name__ == '__main__':
    ### CONFIGURATION
    mode            = 'mc'
    data_handle     = 'dev_level7_mu_e_tau_oscweight_newfeats'
    graph_handle    = 'track_cascadev2'
    db_file         = r'X:\speciale\data\raw\%s\data\%s.db'%(data_handle,data_handle)
    events_path     = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\selection\track_cascade_labelsv2.csv'  #Generate_Event_Array(r'X:\speciale\data\raw\%s'%data_handle,db_file,mode) #r'X:\speciale\data\export\Upgrade Cleaning\selections\martin_selection.csv'#    # r'X:\speciale\data\raw\dev_level2_classification_corsika_genie_muongun_000\meta\1mio_corsika_genie_events.csv' #            
    scalers         = r'X:\speciale\data\raw\%s\meta\transformers.pkl'%data_handle
    path            = r'X:\speciale\data\graphs\%s\%s'%(data_handle,graph_handle)
    n_events_stop   = 2000000
    n_events_start  = 0
    all_events      = pd.read_csv(events_path).loc[:,'event_no'].reset_index(drop=True)
    events_array    = pd.read_csv(events_path).loc[:,'event_no'][n_events_start:n_events_stop].reset_index(drop=True)
    #events_array   = events_array[events_array != 155635386.0]
    classification  = True
    ensemble        = False
    
    fullgrid        = False
    SRT             = False
    GRU             = False
    n_bins          = 50
    upgrade         = False
    file_size       = 50000
    list_break      = 1000
    run_id          = 0
    print_verbose   = 10000
    scaler          = None #FitScaler(db_file,events_array,path,upgrade)
    
    ### AUXILLIARY CONFIGURATION
    n_workers = 5
    
    #feats = str('x,y,z,time,charge_log10')
    feats = ['dom_x','dom_y','dom_z','dom_time','dom_charge','dom_string','dom_pmt','dom_om','dom_lc',
                     'dom_atwd','dom_fadc']
    
    if ensemble == True:
       events_array =  pd.read_csv(r'X:\speciale\data\graphs\%s\%s\predictions\events.csv'%(data_handle,graph_handle)).loc[:,'event_no'][0:].reset_index(drop=True)
    ### DON'T CHANGE THIS
       low_weights  = CreateWeights(n_bins,
                                 db_file,
                                 scalers,
                                 events_array,
                                 data_handle,
                                 graph_handle,
                                 x_low = 1,
                                 type = 'inverse_count',
                                 mode = mode)[0:len(events_array)]
        
       high_weights = CreateWeights(n_bins,
                                     db_file,
                                     scalers,
                                     events_array,
                                     data_handle,
                                     graph_handle,
                                     x_low = 2.5,
                                     type = 'high',
                                     mode = mode)[0:len(events_array)]
    if ensemble == False:
        low_weights  = CreateWeights(n_bins,
                                     db_file,
                                     scalers,
                                     events_array,
                                     data_handle,
                                     graph_handle,
                                     x_low = 1,
                                     type = 'low',
                                     mode = mode)[0:len(events_array)]
        
        high_weights = CreateWeights(n_bins,
                                     db_file,
                                     scalers,
                                     events_array,
                                     data_handle,
                                     graph_handle,
                                     x_low = 2.5,
                                     type = 'high',
                                     mode = mode)[0:len(events_array)]
    #CreateScalers(db_file, graph_handle, data_handle, feats)
    settings = list()
    shuffle_this = np.array([events_array,low_weights,high_weights]).T
    #np.random.shuffle(shuffle_this)  ## DANGEROUS
    event_list = np.array_split(shuffle_this,n_workers)
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
                               print_verbose,
                               GRU,
                               SRT,
                               upgrade,
                               classification,
                               mode,
                               data_handle,
                               ensemble,
                               graph_handle]))
    start = time.time()
    #MakeGraph(settings[0])
    p = Pool(processes = len(settings))
    async_result = p.map_async(MakeGraph, settings)
    p.close()
    p.join()
    print("Complete")
    end = time.time()
    print('total time (s)= ' + str(end-start))

