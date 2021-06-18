import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

        

mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'


scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')




with sqlite3.connect(mc_db) as con:
    query = 'select event_no, zenith from truth where abs(pid) = 14'
    data = pd.read_sql(query,con)
    
    
data['zenith'] = scalers['truth']['zenith'].inverse_transform(np.array(data['zenith']).reshape(-1,1))
#%%
N, bins, patches = plt.hist(data['zenith'], bins=100)

event_nos_bins = []
event_nos = []
threshold = 10998

for i in range(1,len(bins)):
    index = (data['zenith']>= bins[i-1]) & (data['zenith']<= bins[i])
    event_nos_bins.append(data['event_no'][index])


for i in range(0,len(event_nos_bins)):
    if len(event_nos_bins[i]) > threshold:
        sample = event_nos_bins[i].sample(threshold)
        
    else:
        sample = event_nos_bins[i]
    event_nos.extend(sample)
#%%
with sqlite3.connect(mc_db) as con:
    query = 'select event_no, zenith from truth where event_no in %s'%(str(tuple(event_nos)))
    data = pd.read_sql(query,con)
    
    
data['zenith'] = scalers['truth']['zenith'].inverse_transform(np.array(data['zenith']).reshape(-1,1))    

plt.hist(data['zenith'], bins=100)

event_nos = pd.DataFrame(event_nos)
event_nos.columns = ['event_no']
event_nos.to_csv(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\selection\1mio_muon_even_zenith.csv')