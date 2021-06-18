import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

def Width(data,variable,predictor, scalers):
    if variable == 'zenith':
        const = 360/(np.pi*2)
    data[variable] = scalers['truth'][variable].inverse_transform(np.array(data[variable]).reshape(-1,1))*const
    data[predictor] = scalers['truth'][variable].inverse_transform(np.array(data[predictor]).reshape(-1,1))*const
    num_bins = np.arange(0,4.5,0.25)
    E = data['energy_log10']
    fig = plt.figure()
    n, E_bins, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
    plt.close()
    widths = []
    means = []
    for i in range(1,len(E_bins)):
        index = (data['energy_log10'] >= E_bins[i-1]) & (data['energy_log10'] < E_bins[i])
        diff = (data[variable][index] - data[predictor][index]).reset_index(drop = True)
        x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
        print(diff.mean())
        widths.append(np.array((diff[x_75]-diff[x_25])/1.349))
        means.append(data['energy_log10'][index].mean())
    return means,widths

mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'

scalers_dyn = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')
mc_res = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e-check\results.csv')#r'X:\speciale\results\dev_level7_mu_tau_e_retro_000\event_only_level7_all_neutrinos_retro_SRT_4mio\dynedge-E-protov2-zenith\results.csv')

with sqlite3.connect(mc_db) as con:
    query = 'select * from truth where event_no in %s'%(str(tuple(mc_res['event_no'])))
    data = pd.read_sql(query,con)


means, widths = Width(mc_res,'zenith','zenith_pred',scalers_dyn)
plt.errorbar(means,widths,linestyle='dotted',fmt = 'o',capsize = 10)

means, widths = Width(data, 'zenith', 'zenith_retro', scalers_dyn)
plt.errorbar(means,widths,linestyle='dotted',fmt = 'o',capsize = 10)

