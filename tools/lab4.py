import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import scipy as sp
import os
import torch
from torch_geometric.data import DataLoader


def GrabGraphs(path):
    graphs = list()
    for file in os.listdir(path):
        graphs.append(path + '\\' + file)
    if len(graphs) == 0:
        print('FILES NOT FOUND')
    return graphs

def weight(x,x_low,type):
    if type == 'low':
        if x<x_low:
            return 100
    
        if x>= x_low:
            return 1*(1/(1+ x-x_low))
    if type == 'high':
        if(x> x_low):
            return 1
        if(x< x_low):
            return 1*(1/(1+x_low-x))    
data_handle     = 'dev_upgrade_train_step4_001'
graph_handle    = 'event_only_shuffled_input-_target-_bjørn_weights'
db_file         = r'X:\speciale\data\raw\%s\data\%s.db'%(data_handle,data_handle)
scalers         = r'X:\speciale\data\raw\%s\meta\transformers.pkl'%data_handle
#lol = Generate_Event_Array(r'X:\speciale\data\raw\%s'%data_handle,db_file)
events_path     = r'X:\speciale\data\raw\dev_upgrade_train_step4_001\events\events.csv'
events_array    = pd.read_csv(events_path).loc[:,'event_no'].reset_index(drop=True)
n_bins = 50



path = r'X:\speciale\data\graphs\dev_upgrade_train_step4_001\event_only_shuffled_input-_target-_bjørn_weights_wild\train'

graph_files = GrabGraphs(path)
#%%
x = np.arange(-0.5,3,0.001)
w = []
for i in range(0,len(x)):
    w.append(weight(x[i],0.5,'low'))

plt.plot(x,w)


#%%

E  = []
hw = []
lw = []
for graph_file in graph_files:
    data_list = torch.load(graph_file)
    loader = DataLoader(data_list,batch_size = 1024,drop_last=True)
    loader_it = iter(loader)
    for k in range(0,len(loader)):
        graphs = next(loader_it)
        E.extend(graphs.y[:,0])
        hw.extend(graphs.hw)
        lw.extend(graphs.lw)

res = pd.DataFrame(E)
res['lw'] = lw
res.columns = ['E','lw']
res = res.sort_values('E')

#%%
E_plot = []
lw_plot = []
for energy in E:
    E_plot.append(energy.item())
for w in lw:
    lw_plot.append(w.item())

plt.scatter(E_plot,lw_plot)
#%%


low_weights = pd.read_csv(r'X:\speciale\data\graphs\dev_upgrade_train_step4_001\event_only_shuffled_input-_target-_bjørn_weights_test\weights\weights_low.csv')

sca = pd.DataFrame()
scalers = pd.read_pickle(scalers)
scaler_mads = scalers['truth']['energy_log10']                                                             
with sqlite3.connect(db_file) as con:                                                                 
    query = 'select energy_log10 from truth'                                              
    sca = sca.append(pd.read_sql(query, con))
sca = scaler_mads.inverse_transform(np.array(sca).reshape(-1,1)) 

plt.scatter(sca,low_weights.loc[:,low_weights.columns[1]])





 