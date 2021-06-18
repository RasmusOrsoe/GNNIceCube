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
from numba import *
import numba
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
    
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
        fig = plt.figure()
        N, bins, patches = plt.hist(sca, bins = 50)
        plt.close()
        index  = np.digitize(sca, bins=bins)
        #print(index.shape)
        unique_bins = pd.unique(pd.DataFrame(index[:,0]).loc[:,0])
        unique_bins.sort()
        bin_count = []
        for unique_bin in unique_bins:
            bin_count.append(sum(index == unique_bin))
        weights  = []
        print('in')
        for i in range(0,len(sca)):
            weights.append(float(weight(sca[i],x_low,type,bin_count, bins)))
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
        
   #print('weights are set to 1')
   #return 1 
  
    
if __name__ == '__main__':
    ### CONFIGURATION
    mode            = 'mc'
    data_handle     = 'dev_level7_mu_e_tau_oscweight_newfeats'
    graph_handle    = 'whole_sample'
    db_file         = r'X:\speciale\data\raw\%s\data\%s.db'%(data_handle,data_handle)
    events_path     = Generate_Event_Array(r'X:\speciale\data\raw\%s'%data_handle,db_file,mode) #r'X:\speciale\data\export\Upgrade Cleaning\selections\martin_selection.csv'#    # r'X:\speciale\data\raw\dev_level2_classification_corsika_genie_muongun_000\meta\1mio_corsika_genie_events.csv' #            
    scalers         = r'X:\speciale\data\raw\%s\meta\transformers.pkl'%data_handle
    path            = r'X:\speciale\data\graphs\%s\%s'%(data_handle,graph_handle)
    n_events_stop   = 700000
    n_events_start  = 0
    all_events      = pd.read_csv(events_path).loc[:,'event_no'].reset_index(drop=True)
    events_array    = pd.read_csv(events_path).loc[:,'event_no'].sample(frac= 1)[n_events_start:n_events_stop].reset_index(drop=True)
    #events_array   = events_array[events_array != 155635386.0]
    classification  = False
    ensemble        = False
    
    fullgrid        = False
    SRT             = False
    GRU             = False
    n_bins          = 50
    upgrade         = False
    file_size       = 50000
    list_break      = 1000
    run_id          = 1
    print_verbose   = 10000
    scaler          = None #FitScaler(db_file,events_array,path,upgrade)
    
    ### AUXILLIARY CONFIGURATION
    n_workers = 5
    
    

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
    

    with sqlite3.connect(db_file) as con:
        query = 'select zenith from truth'
        zenith = pd.read_sql(query,con)
    zenith = zenith['zenith'][0:70000]
    plt.figure()
    plt.plot(zenith, low_weights[0:70000])