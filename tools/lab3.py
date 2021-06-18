# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:46:11 2020

@author: RahN
"""
import sqlite3
import pandas as pd
import numpy as np

data_handle     = 'dev_numu_train_l5_retro_001'
db_file         = r'J:\speciale\data\raw\%s\data\%s.db'%(data_handle,data_handle)

path = r'J:\speciale\data\raw\%s'%data_handle+'\events\events.csv'
events= pd.read_csv(path).loc[:,'event_no'][0:10]
seq = pd.DataFrame()
sca = pd.DataFrame()
with sqlite3.connect(db_file) as con:                                           #
    query = 'select * from features WHERE event_no IN %s'%(str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
    seq = seq.append(pd.read_sql(query, con))                       
    query = 'select * from truth WHERE event_no IN %s'%(str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
    sca = sca.append(pd.read_sql(query, con))                             #
    cursor = con.cursor()                                                       #
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")