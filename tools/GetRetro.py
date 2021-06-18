# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:49:14 2020

@author: RahN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os
import torch


db_file = r'J:\speciale\data\raw\standard_reco\reco\retro_reco_2.db'


path = r'J:\speciale\data\graphs\dev_numu_train_l5_retro_001\event_only_shuffled(0,1)(1,4)\events\valid'
valid_events = pd.DataFrame()
for file in os.listdir(path):
    data_list_train = torch.load(file)
    loader = DataLoader(data_list_train,batch_size = batch_size,drop_last=True)
    loader_it = iter(loader)
    valid_events = valid_events.append(pd.read_csv(path + '\\' + file))
valid_events = valid_events['event_no'].reset_index(drop = True)    

    
    
with sqlite3.connect(db_file) as con:                                           #
    query = 'select energy_log10, event_no from retro_reco WHERE event_no IN %s '%str(tuple(np.array(valid_events)))
    retro = pd.read_sql(query,con)
    #query = 'select energy_log10 from retro_reco'
    #retro_full = pd.read_sql(query,con)                                        # MERGES ALL .db FILES TO TWO .csv FILES:
    #seq = seq.append(pd.read_sql(query, con))                       
    #query = 'select %s from truth WHERE event_no IN %s'%(truths,str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
    #sca = sca.append(pd.read_sql(query, con))                             #
    cursor = con.cursor()                                                       #
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

data = pd.DataFrame(retro)
data.columns = ['E','event_no']
data.to_csv(r'J:\speciale\data\graphs\dev_numu_train_l5_retro_001\event_only_shuffled(0,1)(1,4)\retro_valid\retro.csv',index = False)