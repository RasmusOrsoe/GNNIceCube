import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import streamlit as st

@st.cache
def setup_stuff():
    db_file = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'
    
    scalers_dyn = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')
    
    results = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e-check\results.csv')
    
    results_E = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-energy-k=8-c3not3-thesis-newlr-30e-check-noprob\results.csv')
    
    results = results.sort_values('event_no').reset_index(drop = True)
    results_E = results_E.sort_values('event_no').reset_index(drop = True)
    results = pd.concat((results,pd.DataFrame(results_E['energy_log10_pred'])), axis = 1).reset_index(drop = True)
    
    events = results['event_no']
    with sqlite3.connect(db_file) as con:
        query = 'select * from truth where event_no in %s'%str(tuple(events))
        data = pd.read_sql(query,con)
    
    data = data.sort_values('event_no').reset_index(drop = True)
    
    variable = 'zenith'
    k_threshold = 0.5
    dynedge = variable + '_pred'
    retro   = variable + '_retro'
    retro_sigma = variable + '_sigma'
    sigma   = dynedge + '_k'
    
    if variable == 'azimuth':
        const = 360/(2*np.pi)
        bins = np.arange(0,6.1,0.1)*const
    if variable == 'zenith':
        const = 360/(2*np.pi)
        bins = np.arange(0,3.1,0.1)*const
    
    plot_data = pd.concat( [pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(results[dynedge]).reshape(-1,1))), 
                               pd.DataFrame(data[retro]),
                               pd.DataFrame(data['energy_log10_retro']),
                               pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(results['energy_log10_pred']).reshape(-1,1))),
                               pd.DataFrame(data[retro_sigma])*const,
                               pd.DataFrame(1/np.sqrt(scalers_dyn['truth'][variable].inverse_transform(np.array(abs(results[sigma])).reshape(-1,1)))),
                               pd.DataFrame(data['osc_weight']),
                               pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(data['energy_log10']).reshape(-1,1))),
                               pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(results[variable]).reshape(-1,1))),
                               pd.DataFrame(data['pid'])],axis = 1)
    
    plot_data.columns = [dynedge,retro,'E_retro','E_pred',retro_sigma,sigma,'osc_weight','energy_log10','zenith','pid']
    plot_data[sigma] = plot_data[sigma]/np.std(plot_data[sigma])
    plot_data = plot_data.loc[abs(plot_data['pid']) == 14,:]
    return plot_data, variable, sigma

plot_data, variable, sigma = setup_stuff()

const  = (360/(2*np.pi))
bins_z = np.arange(0,3.1,0.1)*const
bins_E = np.arange(-0.5,4.1,0.1)
fig_osc_mc, ax_osc = plt.subplots(3,3, figsize = (20,20))
cmap=plt.get_cmap('hot')
fig_osc_mc.suptitle(' %s MC Oscillation'%variable ,size = 20)

ax_osc[0,0].set_ylabel("$cos(ø)_{dyn}$",size = 20)
ax_osc[0,0].set_xlabel('$Log_{10}(E_{dyn})$ [GeV]', size = 20)
ax_osc[0,1].set_ylabel("$cos(ø)_{retro}$",size = 20)
ax_osc[0,1].set_xlabel('$Log_{10}(E_{retro})$ [GeV]', size = 20)
#ax_osc[0,2].set_ylabel("$cos(ø)_{truth}$",size = 20)
ax_osc[0,2].set_xlabel('Zenith [Rad.]', size = 20)

ax_osc[1,0].set_ylabel("$cos(ø)_{dyn}$",size = 20)
ax_osc[1,0].set_xlabel('$Log_{10}(E_{truth})$ [GeV]', size = 20)
ax_osc[1,1].set_ylabel("$cos(ø)_{retro}$",size = 20)
ax_osc[1,1].set_xlabel('$Log_{10}(E_{truth})$ [GeV]', size = 20)
#ax_osc[1,2].set_ylabel("$cos(ø)_{truth}$",size = 20)
ax_osc[1,2].set_xlabel('$Log_{10}(E)$ [GeV]', size = 20)

ax_osc[2,0].set_ylabel("$cos(ø)_{truth}$",size = 20)
ax_osc[2,0].set_xlabel('$Log_{10}(E_{dyn})$ [GeV]', size = 20)
ax_osc[2,1].set_ylabel("$cos(ø)_{truth}$",size = 20)
ax_osc[2,1].set_xlabel('$Log_{10}(E_{retro})$ [GeV]', size = 20)
ax_osc[2,2].set_ylabel("$cos(ø)_{truth}$",size = 20)
ax_osc[2,2].set_xlabel('$Log_{10}(E_{truth})$ [GeV]', size = 20)


full_data = deepcopy(plot_data)

i = st.sidebar.slider('% Most Certain', min_value = 0.0, max_value = 1.0,value = 1.0, step = 0.01)
cmap=plt.get_cmap('hot')
k = i
binsy = np.arange(-1,1,0.01)
binsx = np.arange(-0.5,3.5,0.01)
fig_osc_mc.suptitle(' %s MC Oscillation: %s percent '%(variable, round(i,2)) ,size = 20)
n = int(np.ceil(i*len(full_data)))
plot_data = full_data.nsmallest(n, sigma).reset_index(drop = True)
cos = np.cos(plot_data['zenith'])
cos_dyn = np.cos(plot_data['zenith_pred'])
cos_retro = np.cos(plot_data['zenith_retro'])
E_dyn = plot_data['E_pred']
E_retro = plot_data['E_retro']
E = plot_data['energy_log10']
weights = plot_data['osc_weight']
ax_osc[0,0].hist2d(E_dyn.squeeze(),cos_dyn.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy], cmap = cmap)
ax_osc[0,1].hist2d(E_retro.squeeze(),cos_retro.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)
#ax_osc[0,2].hist2d(E,cos.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)

ax_osc[1,0].hist2d(E.squeeze(),cos_dyn.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)
ax_osc[1,1].hist2d(E.squeeze(),cos_retro.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)
#ax_osc[1,2].hist2d(E,cos.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)

ax_osc[2,0].hist2d(E_dyn.squeeze(),cos.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)
ax_osc[2,1].hist2d(E_retro.squeeze(),cos.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)
ax_osc[2,2].hist2d(E,cos.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)

#fig_dist, ax = plt.subplots(2,2, figsize = (20,10))

ax_osc[0,2].hist(plot_data['zenith']*const, bins = bins_z, label = 'truth', color = 'lightgrey') 
ax_osc[0,2].hist(plot_data['zenith_pred']*const, bins = bins_z,histtype = 'step', label = 'dynedge')
ax_osc[0,2].hist(plot_data['zenith_retro']*const, bins = bins_z,histtype = 'step', label = 'retro')
ax_osc[0,2].legend()  
#ax[0,1].hist(full_data['zenith']*const, bins = bins_z, label = 'truth', color = 'lightgrey') 
#ax[0,1].hist(full_data['zenith_pred']*const, bins = bins_z,histtype = 'step', label = 'dynedge')
#ax[0,1].hist(full_data['zenith_retro']*const, bins = bins_z,histtype = 'step', label = 'retro')
#ax[0,1].legend()  

ax_osc[1,2].hist(plot_data['energy_log10'], bins = bins_E, label = 'truth', color = 'lightgrey') 
ax_osc[1,2].hist(plot_data['E_pred'], bins = bins_E,histtype = 'step', label = 'dynedge')
ax_osc[1,2].hist(plot_data['E_retro'], bins = bins_E,histtype = 'step', label = 'retro')
ax_osc[1,2].legend()  
#ax[1,1].hist(full_data['energy_log10'], bins = bins_E, label = 'truth', color = 'lightgrey') 
#ax[1,1].hist(full_data['E_pred'], bins = bins_E,histtype = 'step', label = 'dynedge')
#ax[1,1].hist(full_data['E_retro'], bins = bins_E,histtype = 'step', label = 'retro')
#ax[1,1].legend()  



st.pyplot(fig_osc_mc)
#st.pyplot(fig_dist)






