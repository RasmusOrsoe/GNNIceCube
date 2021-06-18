import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from copy import deepcopy
#%%
db_file = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'

#scalers_retro = pd.read_pickle(r'X:\speciale\data\rawdev_numu_train_l5_retro_001\meta\transformers.pkl')

scalers_dyn = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')

results = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\2mio_muons_only\dynedge-protov2-zenith-k=8-c3not3-extrafeats\results.csv')

results_E = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\2mio_muons_only\dynedge-protov2-energy-k=8-c3not3-extrafeats-noprob\results.csv')

#coln = ['event_no', 'zenith_pred', 'zenith_k','E_pred']

results = results.sort_values('event_no').reset_index(drop = True)
results_E = results_E.sort_values('event_no').reset_index(drop = True)
results = pd.concat((results,pd.DataFrame(results_E['energy_log10_pred'])), axis = 1).reset_index(drop = True)


#results = pd.concat((results,pd.DataFrame(results_E['energy_log10_pred_sigma'])), axis = 1)

#results.columns = coln



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
#else:
#    const = 1
#    bins = None

plot_data = pd.concat( [pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(results[dynedge]).reshape(-1,1))), 
                           pd.DataFrame(data[retro]),
                           pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(data['energy_log10_retro']).reshape(-1,1))),
                           pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(results['energy_log10_pred']).reshape(-1,1))),
                           pd.DataFrame(data[retro_sigma])*const,
                           pd.DataFrame(1/np.sqrt(scalers_dyn['truth'][variable].inverse_transform(np.array(results[sigma]).reshape(-1,1)))),
                           pd.DataFrame(data['osc_weight']),
                           pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(data['energy_log10']).reshape(-1,1))),
                           pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(results[variable]).reshape(-1,1)))],axis = 1)

plot_data.columns = [dynedge,retro,'E_retro','E_pred',retro_sigma,sigma,'osc_weight','energy_log10','zenith']
#plot_data_real = plot_data_real.sort_values(retro)

fig_osc_mc, ax_osc = plt.subplots(3, 1)
cmap=plt.get_cmap('hot')
fig_osc_mc.suptitle(' %s MC Oscillation'%variable ,size = 20)
energy = plot_data['energy_log10']
#zenith = scalers['zenith'].inverse_transform(np.array(data_osc['zenith']).reshape(-1,1))
cos_dyn = np.cos(plot_data[dynedge]/const)#np.cos(plot_data[dynedge]/const)
cos_retro = np.cos(plot_data[retro]/const)
cos = np.cos(plot_data[variable]/const)
weights = plot_data['osc_weight']
ax_osc[0].hist2d(energy.squeeze(),cos_dyn.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)
ax_osc[1].hist2d(energy.squeeze(),cos_retro.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)

ax_osc[2].hist2d(energy.squeeze(),cos.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)

#plt.xlabel("$Energy_{log10}$",size = 20)
ax_osc[0].set_title('No Cut')
ax_osc[2].set_ylabel("$cos(ø)_{true}$",size = 20)
ax_osc[0].set_ylabel("$cos(ø)_{dynedge}$",size = 20)
ax_osc[1].set_ylabel("$cos(ø)_{retro}$",size = 20)
ax_osc[2].set_xlabel('$Log_{10}(E_{True})$ [GeV]', size = 20)
#plt.colorbar()


fig_osc_mc, ax_osc = plt.subplots(1, 3)
cmap=plt.get_cmap('hot')
fig_osc_mc.suptitle(' %s MC Oscillation'%variable ,size = 20)
cos = np.cos(plot_data['zenith']/const)
#zenith = scalers['zenith'].inverse_transform(np.array(data_osc['zenith']).reshape(-1,1))
E_dyn = plot_data['E_pred']#np.cos(plot_data[dynedge]/const)
E_retro = plot_data['E_retro']
E = plot_data['energy_log10']
weights = plot_data['osc_weight']
ax_osc[0].hist2d(E_dyn.squeeze(),cos.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)
ax_osc[1].hist2d(E_retro.squeeze(),cos.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)

ax_osc[2].hist2d(E,cos.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)

#plt.xlabel("$Energy_{log10}$",size = 20)
ax_osc[0].set_title('No Cut')
ax_osc[2].set_ylabel("$cos(ø)_{true}$",size = 20)
ax_osc[0].set_ylabel("$cos(ø)_{true}$",size = 20)
ax_osc[1].set_ylabel("$cos(ø)_{true}$",size = 20)
ax_osc[2].set_xlabel('$Log_{10}(E_{True})$ [GeV]', size = 20)

#%%
fig_osc_mc, ax_osc = plt.subplots(1, 3, figsize = (20,10))
cmap=plt.get_cmap('hot')
fig_osc_mc.suptitle(' %s MC Oscillation'%variable ,size = 20)
ax_osc[0].set_ylabel("$cos(ø)_{dyn}$",size = 20)
ax_osc[0].set_xlabel('$Log_{10}(E_{dyn})$ [GeV]', size = 20)

ax_osc[1].set_ylabel("$cos(ø)_{retro}$",size = 20)
ax_osc[1].set_xlabel('$Log_{10}(E_{retro})$ [GeV]', size = 20)

ax_osc[2].set_ylabel("$cos(ø)_{truth}$",size = 20)
ax_osc[2].set_xlabel('$Log_{10}(E_{truth})$ [GeV]', size = 20)

full_data = deepcopy(plot_data)
def animate(i):
    
    cmap=plt.get_cmap('hot')
    k = np.arange(0.1,1.1,0.1)
    binsy = np.arange(-1,1,0.01)
    binsx = np.arange(-0.5,3.5,0.01)
    fig_osc_mc.suptitle(' %s MC Oscillation: %s percent '%(variable, round(k[i],2)) ,size = 20)
    n = int(np.ceil(k[i]*len(full_data)))
    plot_data = full_data.nsmallest(n, sigma).reset_index(drop = True)
    cos = np.cos(plot_data['zenith'])
    cos_dyn = np.cos(plot_data['zenith_pred'])
    cos_retro = np.cos(plot_data['zenith_retro'])
    E_dyn = plot_data['E_pred']
    E_retro = plot_data['E_retro']
    E = plot_data['energy_log10']
    weights = plot_data['osc_weight']
    ax_osc[0].hist2d(E_dyn.squeeze(),cos_dyn.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)
    ax_osc[1].hist2d(E_retro.squeeze(),cos_retro.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)
    ax_osc[2].hist2d(E,cos.squeeze(), weights = weights.squeeze(), bins = [binsx,binsy],cmap = cmap)

ani = FuncAnimation(fig_osc_mc, animate, interval=1)
plt.show()