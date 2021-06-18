import numpy as np
import pandas as pd
import sqlite3
import torch
from sklearn.preprocessing import MinMaxScaler
import os

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



data_handle     = 'dev_numu_train_upgrade_step4_2020_00'
graph_handle    = 'event_only_shuffled(0,1)(1,4)_exp_padded'
db_file         = r'X:\speciale\data\raw\%s\data\%s.db'%(data_handle,data_handle)

CreateScalers(db_file,graph_handle,data_handle,['dom_x'])