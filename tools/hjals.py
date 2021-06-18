import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

path = r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything_wtest\dynedge-protov2-energy-k=8-c3not3-w_test_val-noprob\results.csv'


data = pd.read_csv(path)

db_file = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'
scalers = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl'

scalers = pd.read_pickle(scalers)['truth']['energy_log10']

with sqlite3.connect(db_file) as con:
    query = 'select * from truth where event_no in %s'%str(tuple(data['event_no']))
    truth = pd.read_sql(query,con)
    
truth['energy_log10'] = scalers.inverse_transform(np.array(truth['energy_log10']).reshape(-1,1))
data['energy_log10_pred'] = scalers.inverse_transform(np.array(data['energy_log10_pred']).reshape(-1,1))
bins = np.arange(-3,3, 0.1)
fig = plt.figure()
#plt.hist(np.arctan2(data['energy_log10'], data['energy_log10_pred']), histtype = 'step', label='atan2 pred', bins = bins)
#plt.hist(np.arccos(data['cos_zenith']), label = 'arccos pred')
plt.hist(truth['energy_log10'], histtype = 'step', label = 'truth', bins = bins)
plt.hist(data['energy_log10_pred'], histtype = 'step', label = 'pred', bins = bins)
plt.hist(truth['energy_log10_retro'], histtype = 'step', label = 'retro', bins = bins)
plt.legend()
#fig = plt.figure()
#plt.hist(np.arctan2(data['sin_azimuth'], data['cos_azimuth']), histtype = 'step', label='pred')
#plt.hist(data['azimuth'], histtype = 'step', label = 'truth')
#plt.legend()
#%%
fig = plt.figure()
plt.hist2d(data['cos_zenith'], np.cos(data['zenith']), bins = 30)