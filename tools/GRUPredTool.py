import numpy as np
import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt


def grab_files(path):
    files = os.listdir(path)
    
    results = pd.DataFrame()
    for file in files:
        results = results.append(pd.read_csv(path + '\\' + file))
    
    
    
    return results

path = r'X:\speciale\ice-gru-master\predictions\dev_numu_train_l2_2020_01\high_energy\files'

db_file = r'X:\speciale\data\raw\dev_numu_train_l2_2020_01\data\dev_numu_train_l2_2020_01.db'

res = grab_files(path).loc[:,['event_no','true_primary_energy']]

events = res.loc[:,'event_no']
            
scaler = pd.read_pickle(r'X:\speciale\data\raw\dev_numu_train_l2_2020_01\meta\transformers.pkl')['truth']['energy_log10']


with sqlite3.connect(db_file) as con:  
    query = 'select energy_log10 from truth WHERE event_no IN %s'%(str(tuple(events)))
    E = pd.read_sql(query, con)          

res = res.reset_index(drop = True)
res = pd.concat((res,E),axis = 1)

res.columns = ['event_no','E_pred','E']

res = res.loc[~np.isnan(res['E_pred']),:]

res['E_pred'] = scaler.transform(np.array(res['E_pred']).reshape(-1,1))


res.to_csv(r'X:\speciale\ice-gru-master\predictions\dev_numu_train_l2_2020_01\high_energy\results.csv',index = False)

