import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from multiprocessing import Pool
import multiprocessing
import time
import os

def Aggregate(path):
    data = pd.DataFrame()
    for file in os.listdir(path):
        data = data.append(pd.read_csv(path + '\\' + file))
    return data



path_clean = r'X:\speciale\data\export\Upgrade Cleaning\distributions\automated\old\clean_srt'
path_dirty = r'X:\speciale\data\export\Upgrade Cleaning\distributions\automated\old\dirty_srt'

data_clean = Aggregate(path_clean)
data_dirty = Aggregate(path_dirty)

datas = [data_clean,data_dirty]

##########################
# DOM130 + DOM120 vs DOM20
##########################
type_count = 0
figs = list()
for data in datas:
    if type_count == 0:
        handle = 'clean'
    else:
        handle = 'dirty'
    fig = plt.figure()
    plt.title('dom120 + dom130 vs dom20 | " %s "'%handle)    
    cut_off_dom20 = 50
    cut_off_y     = 100
    
    n_dom120_dom130 = (data['dom130'] + data['dom120'])[~np.isnan(data['dom130'] + data['dom120'])]
    n_dom20 = data['dom20'][~np.isnan(data['dom20'])]
    
    ## CUT_OFF
    n_dom120_dom130 = n_dom120_dom130[n_dom20<cut_off_dom20]
    n_dom20  = n_dom20[n_dom20<cut_off_dom20][n_dom120_dom130<cut_off_y]
    n_dom120_dom130 = n_dom120_dom130[n_dom120_dom130<cut_off_y]
    ## PLOT
    plt.hist2d(n_dom20,n_dom120_dom130,bins=50,label=handle)
    type_count += 1
    plt.savefig(r'X:\speciale\data\export\Upgrade Cleaning\for meeting\dom_check_%s.pdf'%handle, papertype= 'a4')
    figs.append(fig)
    #plt.close(fig)
#%%
##########################
# VERTEX_Z VS R
##########################
data_handle         = 'dev_upgrade_train_step4_001'
db_file             = r'X:\speciale\data\raw\%s\data\%s.db'%(data_handle,data_handle)
scaler_path         = r'X:\speciale\data\raw\%s\meta\transformers.pkl'%data_handle
scalers             = pd.read_pickle(scaler_path)
scaler_x            = scalers['truth']['position_x']
scaler_y            = scalers['truth']['position_y']
scaler_z            = scalers['truth']['position_z']

fig = plt.figure()
data = data_dirty
events = data['event_no'][~np.isnan(data['event_no'])].astype(int).reset_index(drop = True)
with sqlite3.connect(db_file) as con:          
    query = 'select position_x,position_y,position_z from truth WHERE event_no IN %s'%(str(tuple(events)))                                              # THESE ARE THEN WRITTEN TO DRIVE
    sca = pd.read_sql(query, con)                             #
    cursor = con.cursor()

sca.loc[:,'position_x'] = scaler_x.inverse_transform(np.array(sca.loc[:,'position_x']).reshape(-1,1))
sca.loc[:,'position_y'] = scaler_x.inverse_transform(np.array(sca.loc[:,'position_y']).reshape(-1,1))
sca.loc[:,'position_z'] = scaler_x.inverse_transform(np.array(sca.loc[:,'position_z']).reshape(-1,1))

r = np.sqrt(np.array(sca['position_x']**2 + sca['position_y']**2))

plt.hist2d(r,sca['position_z'],bins =  100)
plt.ylim(-250,250)
plt.xlim(0,500)

