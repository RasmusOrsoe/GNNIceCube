import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np


mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'

scalers_dyn = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')

mc_res = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e-check\results.csv')#r'X:\speciale\results\dev_level7_mu_tau_e_retro_000\event_only_level7_all_neutrinos_retro_SRT_4mio\dynedge-E-protov2-zenith\results.csv')


doms = []
event_nos = mc_res['event_no']

n_jobs = np.array_split(event_nos,1000)
k = 0
for job in n_jobs:
    print(k)
    event_nos = job
    with sqlite3.connect(mc_db) as con:
        query = 'select event_no from features where event_no in %s'%str(tuple(event_nos))
        data = pd.read_sql(query,con)
    
    
    for event in event_nos:
        doms.append([event,sum(data['event_no'] == event)])
    k+=1
doms = pd.DataFrame(doms)
doms.columns = ['event_no', 'n_doms']
doms.to_csv(r'X:\speciale\resources\n_doms.csv')

#strings = []
#positions = []
#for event in event_nos:
#    positions = []
#    mid = data.loc[data['event_no']==event,:].reset_index(drop = True)
#    for k in range(0,len(mid)):
#        pos = tuple(np.array(mid.loc[k,['dom_x','dom_y']]))
#        positions.append(pos)
#        
#    strings.append([event,len(pd.unique(positions))])
#        
#        
#res = pd.DataFrame(strings)
#res.columns = ['event_no', 'n_strings']

#res.to_csv(r'X:\speciale\resources\n_strings.csv')
    