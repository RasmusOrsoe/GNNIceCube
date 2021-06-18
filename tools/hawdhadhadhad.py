import sqlite3
import pandas as pd
import numpy as np

scaler1 = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_tau_e_muongun_classification\meta\transformers.pkl')['features']['width']
path2 = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'

with sqlite3.connect(path2) as con:
    query = 'select event_no, pid, interaction_type  from truth where abs(pid) == 14 and interaction_type == 1'
    tracks = pd.read_sql(query,con)

with sqlite3.connect(path2) as con:
    query = 'select event_no, pid, interaction_type  from truth where event_no not in %s'%(str(tuple(tracks['event_no'])))
    cascade = pd.read_sql(query,con)



labels = tracks.append(cascade, ignore_index = True).reset_index(drop = True)

#labels = labels.sample(frac = 0.15).reset_index(drop = True)

labels.to_csv(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\selection\track_cascade_labels.csv')


import matplotlib.pyplot as plt

fig = plt.figure()
plt.hist(cascade['interaction_type'])
plt.title('raw')

fig = plt.figure()
plt.hist(abs(cascade['pid']))
plt.title('raw')


cascade_pid14 = cascade.loc[abs(cascade['pid'] == 14),:]
cascade_pid12 = cascade.loc[abs(cascade['pid'] == 12),:].sample(len(cascade_pid14))
cascade_pid16 = cascade.loc[abs(cascade['pid'] == 16),:].sample(len(cascade_pid14))

cascade_subsample = cascade_pid14.append(cascade_pid12).append(cascade_pid16)



cascade_it2 = cascade_subsample.loc[cascade['interaction_type'] == 2,:]
cascade_it1 = cascade_subsample.loc[cascade['interaction_type'] != 2,:].sample(len(cascade_it2))




cascade_subsubsample = cascade_it2.append(cascade_it1)

fig = plt.figure()
plt.hist(cascade_subsample['interaction_type'])
plt.title('it cut')

fig = plt.figure()
plt.hist(abs(cascade_subsample['pid']))
plt.title('it cut')

fig = plt.figure()
plt.hist(cascade_subsubsample['interaction_type'])
plt.title('pid cut')

fig = plt.figure()
plt.hist(abs(cascade_subsubsample['pid']))
plt.title('pid cut')


tracks = tracks.sample(len(cascade_subsubsample))
labels = tracks.append(cascade_subsubsample)
labels.to_csv(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\selection\track_cascade_labelsv2.csv')