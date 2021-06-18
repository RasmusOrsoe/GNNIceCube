import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

db_file = r'X:\speciale\data\raw\oscnext_IC8611_newfeats_000_mc_scaler\data\oscnext_IC8611_newfeats_000_mc_scaler.db'

mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'

#scalers_retro = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_oscNext_IC86_11_with_retro\meta\transformers.pkl')

scalers_dyn = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')

results = pd.read_csv(r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler\regression\dynedge-protov2-energy-k=8-c3not3-thesis-oldlr-50e-check-noprob\results.csv')#r'X:\speciale\results\dev_level7_oscNext_IC86_003\event_only_level7_all_oscNext_IC86_003\dynedge-E-protov2-zenith\results.csv')

mc_res = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-energy-k=8-c3not3-thesis-newlr-30e-check-noprob\results.csv')#r'X:\speciale\results\dev_level7_mu_tau_e_retro_000\event_only_level7_all_neutrinos_retro_SRT_4mio\dynedge-E-protov2-zenith\results.csv')

labels = pd.read_csv(r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler\regression\dynedge-protov2-classification-k=8-c3not3-classification-even\labels.csv')
labels = labels.sort_values('event_no').reset_index(drop = True)
strings = pd.read_csv(r'X:\speciale\resources\n_doms.csv').sort_values('event_no').reset_index(drop = True)
results = results.sort_values('event_no').reset_index(drop = True)
events = results['event_no']
with sqlite3.connect(db_file) as con:
    query = 'select * from truth where event_no in %s'%str(tuple(events))
    data = pd.read_sql(query,con)
    query = 'select * from features where event_no in %s'%str(tuple(events))
    feats = pd.read_sql(query,con)

data = data.sort_values('event_no').reset_index(drop = True)

variable = 'energy_log10'
k_threshold = 0.50
dynedge = variable + '_pred'
retro   = variable + '_retro'
retro_sigma = variable + '_sigma'
sigma   = dynedge + '_sigma'

if variable == 'azimuth':
    const = 360/(2*np.pi)
    bins = np.arange(0,6.1,0.1)*const
if variable == 'zenith':
    const = 360/(2*np.pi)
    bins = np.arange(0,3.1,0.1)*const
if variable == 'energy_log10':
    const = 1
    bins = np.arange(0,4,0.1)


plot_data_real = pd.concat( [pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(results[dynedge]).reshape(-1,1))*const), 
                           pd.DataFrame(data[retro])*const,
                           pd.DataFrame(data[retro_sigma])*const,
                           pd.DataFrame(labels['neutrino'])],axis = 1)

plot_data_real.columns = [dynedge,retro,retro_sigma, 'neutrino']
plot_data_real.to_csv(r'X:\speciale\litterature\to_troels\real_data_energy_log10.csv')
plot_data_real = plot_data_real.sort_values(retro)



fig, ax = plt.subplots(4, 1)
fig.suptitle('IC86.11 OscNext Real Data: %s predictions'%variable,size = 20)
neutrino_index = plot_data_real['neutrino'] == 1
ax[0].hist(plot_data_real[retro],bins = bins,label = 'retro')
ax[0].hist(plot_data_real[dynedge],histtype='step',bins = bins,label = 'dynedge')
ax[0].set_title('No Cut', size = 20)

ax[0].legend()
ax[0].set_title('No Cut', size = 20)

ax[1].hist2d(plot_data_real[retro],plot_data_real[dynedge],bins = bins)
ax[1].set_ylabel('dynedge',size =  20)
ax[2].hist2d(plot_data_real[retro][neutrino_index],plot_data_real[dynedge][neutrino_index], bins = bins)
ax[2].set_ylabel('dynedge', size = 20)

p_x = plot_data_real[retro].mean()
p_y = plot_data_real[dynedge].max() - 4/5*plot_data_real[dynedge].max()
ax[1].text(p_x,p_y, 'Full IC86.11 OscNext', fontsize=15,  color='red')
ax[2].text(p_x,p_y, '$\\nu$-classified IC86.11 OscNext', fontsize=15,  color='red')


fig_sigma, ax_sigma = plt.subplots(2, 2)
fig_sigma.suptitle('IC86.11 OscNext Real Data: %s $\sigma$'%variable,size = 20)
#not_nan = ~np.isnan(plot_data_real[sigma])
#ax_sigma[0,0].hist(plot_data_real[retro_sigma][not_nan], label =  'retro', histtype = 'step', bins = np.arange(0,5,0.2))
#ax_sigma[0,0].hist(plot_data_real[sigma][not_nan], label =  'dynedge', histtype = 'step',  bins = np.arange(0,5,0.2))
#ax_sigma[0,0].set_title('No Cut', size = 20)
#ax_sigma[0,0].set_xlabel('$\sigma$ [Deg.]', size = 20)
#ax_sigma[1,0].hist2d(plot_data_real[retro_sigma][not_nan],plot_data_real[sigma][not_nan], bins = np.arange(0,5,0.1))
#ax_sigma[1,0].set_xlabel('$retro_{\sigma}$ ', size = 20)
#ax_sigma[1,0].set_ylabel('$dynedge_{\sigma}$ ', size = 20)
#ax_sigma[0,0].set_title('No Cut', size = 20)
#ax_sigma[0,0].legend()


####
#
# Real K cut
#
#####

#ax[0,1].hist(plot_data_real[retro],bins = bins,label = 'retro')
#ax[0,1].hist(plot_data_real[dynedge],histtype='step',bins = bins,label = 'dynedge')
#ax[0,1].set_title('$K < %s \cdot k_{mean}$'%k_threshold, size = 20)

#ax[0,1].legend()
#ax[0,1].set_title('50 %s Best Uncertainties Dynedge'%chr(37), size = 20)

#ax[1,1].hist2d(plot_data_real[retro],plot_data_real[dynedge],bins = bins)
#ax[1,1].text(p_x,p_y, 'IC86.11 OscNext', fontsize=15,  color='red')

#not_nan = ~np.isnan(plot_data_real[sigma])
#ax_sigma[0,1].hist(plot_data_real[retro_sigma][not_nan], label =  'retro', histtype = 'step', bins = np.arange(0,5,0.2))
#ax_sigma[0,1].hist(plot_data_real[sigma][not_nan], label =  'dynedge', histtype = 'step',  bins = np.arange(0,5,0.2))
#ax_sigma[0,1].set_xlabel('$\sigma$ ', size = 20)
#ax_sigma[0,1].set_title('50 %s Most Certain Dynedge'%chr(37), size = 20)
#ax_sigma[1,1].hist2d(plot_data_real[retro_sigma][not_nan],plot_data_real[sigma][not_nan], bins = np.arange(0,5,0.1))
#ax_sigma[1,1].set_xlabel('$retro_{\sigma}$ [Deg.]', size = 20)
#ax_sigma[1,1].set_ylabel('$dynedge_{\sigma}$ [Deg.]', size = 20)
#ax_sigma[0,1].legend()



##### MC
mc_res =  mc_res.sort_values('event_no').reset_index(drop = True)
mc_events = mc_res['event_no']
#r'X:\speciale\data\raw\standard_reco\reco\retro_reco_2.db'
with sqlite3.connect(mc_db) as con:
    query = 'select * from truth where event_no in %s'%str(tuple(mc_events))
    retro_mc = pd.read_sql(query,con)
    

retro_mc = retro_mc.sort_values('event_no').reset_index(drop = True)

ab = mc_res
bc = retro_mc

plot_data = pd.concat( [pd.DataFrame(mc_res['event_no']),
                        pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[dynedge]).reshape(-1,1))*const),
                        pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[variable]).reshape(-1,1))*const),
                        pd.DataFrame(retro_mc[retro]*const),
                        pd.DataFrame(strings['n_doms'])],axis = 1)
                        #pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(mc_res['energy_log10']).reshape(-1,1)))],axis = 1)
cd = plot_data

plot_data.columns = ['event_no',dynedge,variable,retro,'n_doms']
plot_data = plot_data.reset_index(drop = True)



ax[3].hist2d(plot_data[retro],plot_data[dynedge],bins = bins)
ax[3].set_ylabel('dynedge',size =  20)
ax[3].set_xlabel('retro',size =  20)
p_x = plot_data[retro].mean()
p_y = plot_data[dynedge].max() - 2/5*plot_data[dynedge].max()
ax[3].text(p_x,p_y, 'MC' +  ' ($v_{\mu, e, \\tau}$)', fontsize=20,  color='red')


###### 
#
# PLOT ERROR
#
######


result = plot_data.sort_values('n_doms')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['n_doms'].reset_index(drop = True)
num_bins = 25
fig3 = plt.figure()
n, bins_e, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
plt.close()
means_E = list()
medians_error = list()
medians_error_retro = list()
errors_width = list()
errors_width_retro = list()
for k in range(len(bins_e)-1):
    index = (E >= bins_e[k]) & (E <= bins_e[k+1])
    if(sum(index) != 0):
        means_E.append(np.mean(E[index]))
        medians_error.append(np.median(pred[index]-true[index]))
        medians_error_retro.append(np.median(retro_pred[index]-true[index]))
        diff = (pred - true)[index].reset_index(drop = True)
        diff_retro = (retro_pred - true)[index].reset_index(drop = True)
        x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
        x_25_retro = abs(diff_retro-np.percentile(diff_retro,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75_retro = abs(diff_retro-np.percentile(diff_retro,75,interpolation='nearest')).argmin() #int(0.84*N)
        
        N = sum(index)
        fe_25 = sum(diff <= diff[x_25])/N
        fe_75 = sum(diff <= diff[x_75])/N
        errors_width.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25**2 + 1/fe_75**2))*(1/1.349))
        
        fe_25_retro = sum(diff_retro <= diff_retro[x_25_retro])/N
        fe_75_retro = sum(diff_retro <= diff_retro[x_75_retro])/N
        errors_width_retro.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25_retro**2 + 1/fe_75_retro**2))*(1/1.349))
        if( k == 0):
            errors = np.array([np.median(diff) - diff[x_25], 
                               np.median(diff) - diff[x_75]])
            width = np.array(-diff[x_25]+ diff[x_75])/1.349
            
            errors_retro = np.array([np.median(diff_retro) - diff_retro[x_25_retro], 
                               np.median(diff_retro) - diff_retro[x_75_retro]])
            width_retro = np.array(-diff_retro[x_25_retro]+ diff_retro[x_75_retro])/1.349
        else:
            errors = np.c_[errors,np.array([np.median(diff) - diff[x_25],
                                            np.median(diff) - diff[x_75]])]
            width = np.r_[width,np.array(-diff[x_25]+ diff[x_75])/1.349]
             
            errors_retro = np.c_[errors_retro,np.array([np.median(diff_retro) - diff_retro[x_25_retro],
                                            np.median(diff_retro) - diff_retro[x_75_retro]])]
            width_retro = np.r_[width_retro,np.array(-diff_retro[x_25_retro]+ diff_retro[x_75_retro])/1.349]
    



fig_perf, axs_perf = plt.subplots(2, 2)
fig_perf.suptitle(' %s MC Performance: dynedge NC + CC'%variable,size = 20)


## Dynedge        
axs_perf[0,0].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,0].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,0].legend()
axs_perf[0,0].grid()
axs_perf[0,0].set_title('No Cut', size = 20)
if(variable != 'energy_log10'):
    axs_perf[0,0].set_ylabel('$\Delta ø$ [Deg.]', size = 20)
else:
    axs_perf[0,0].set_ylabel(r'$log(\frac{E_{pred}}{E}$)', size = 20)
          
axs_perf[1,0].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,0].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,0].legend()
axs_perf[1,0].grid()
axs_perf[1,0].set_xlabel('N Pulses',size = 20)
if(variable != 'energy_log10'):
    axs_perf[1,0].set_ylabel('$W(\Delta ø) [Deg.]$', size = 20)
    
else:
    axs_perf[1,0].set_ylabel(r'W($log(\frac{E_{pred}}{E}$))', size = 20)
#########



########
fig, axs = plt.subplots(3, 2)

fig.suptitle(' %s MC Distributions'%variable,size = 20)

axs[0, 0].hist(plot_data[retro],bins = bins,label = 'retro')
axs[0, 0].hist(plot_data[variable],bins = bins,label = 'truth',histtype='step')
axs[0, 0].grid()

axs[0, 0].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
axs[0, 0].legend()
axs[0, 0].set_title('No Cut', size = 20)


axs[1, 0].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro')
axs[1, 0].set_ylabel('retro', size = 20)


axs[2, 0].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro')
axs[2, 0].set_xlabel('truth',size = 20)
axs[2, 0].set_ylabel('dynedge', size = 20)
axs[2, 0].legend()



######
#
# K - CUT
#
######
print('Data Before Cut: %s'%len(plot_data))
a = len(plot_data)

#cut = k_threshold*plot_data[sigma].median()

#n = int(len(plot_data)*k_threshold)

#plot_data = plot_data.nsmallest(n, sigma).sort_values('event_no')

print('Data After Cut: %s'%len(plot_data))

print('Retainment: %s procent'%(len(plot_data)/a))


#axs[0, 1].hist(plot_data[retro],bins = bins,label = 'retro')
#axs[0, 1].hist(plot_data[variable],bins = bins,label = 'truth',histtype='step')
#axs[0, 1].grid()
#axs[0, 1].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
#axs[0, 1].legend()
#axs[0, 1].set_title('50 %s Best Uncertainties Dynedge'%chr(37), size = 20)
#axs[1, 1].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro')
#axs[2, 1].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro')
#axs[2, 1].set_xlabel('truth', size = 20)
#axs[2, 1].legend()


#### This is the very first plot with real data in it

ax[3].hist2d(plot_data[retro],plot_data[dynedge],bins = bins)
#ax[2,1].set_ylabel('$predictions_{dynedge}$',size =  20)
ax[3].set_xlabel('retro',size =  20)
#ax[3].text(p_x,p_y, 'MC  ($v_{\mu, e, \\tau}$)', fontsize=20,  color='red')

####


###### 
#
# PLOT ERROR
#
######


result = plot_data.sort_values('n_doms')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['n_doms'].reset_index(drop = True)
num_bins = 25
fig3 = plt.figure()
n, bins_e, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
plt.close()
means_E = list()
medians_error = list()
medians_error_retro = list()
errors_width = list()
errors_width_retro = list()
for k in range(len(bins_e)-1):
    index = (E >= bins_e[k]) & (E <= bins_e[k+1])
    if(sum(index) != 0):
        means_E.append(np.mean(E[index]))
        medians_error.append(np.median(pred[index]-true[index]))
        medians_error_retro.append(np.median(retro_pred[index]-true[index]))
        diff = (pred - true)[index].reset_index(drop = True)
        diff_retro = (retro_pred - true)[index].reset_index(drop = True)
        ylabel = r'$log(\frac{E_{pred}}{E}$)'
        title = r' $log(\frac{E_{pred}}{E}$) vs E'
        x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
        x_25_retro = abs(diff_retro-np.percentile(diff_retro,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75_retro = abs(diff_retro-np.percentile(diff_retro,75,interpolation='nearest')).argmin() #int(0.84*N)
        
        N = sum(index)
        fe_25 = sum(diff <= diff[x_25])/N
        fe_75 = sum(diff <= diff[x_75])/N
        errors_width.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25**2 + 1/fe_75**2))*(1/1.349))
        
        fe_25_retro = sum(diff_retro <= diff_retro[x_25_retro])/N
        fe_75_retro = sum(diff_retro <= diff_retro[x_75_retro])/N
        errors_width_retro.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25_retro**2 + 1/fe_75_retro**2))*(1/1.349))
        if( k == 0):
            errors = np.array([np.median(diff) - diff[x_25], 
                               np.median(diff) - diff[x_75]])
            width = np.array(-diff[x_25]+ diff[x_75])/1.349
            
            errors_retro = np.array([np.median(diff_retro) - diff_retro[x_25_retro], 
                               np.median(diff_retro) - diff_retro[x_75_retro]])
            width_retro = np.array(-diff_retro[x_25_retro]+ diff_retro[x_75_retro])/1.349
        else:
            errors = np.c_[errors,np.array([np.median(diff) - diff[x_25],
                                            np.median(diff) - diff[x_75]])]
            width = np.r_[width,np.array(-diff[x_25]+ diff[x_75])/1.349]
             
            errors_retro = np.c_[errors_retro,np.array([np.median(diff_retro) - diff_retro[x_25_retro],
                                            np.median(diff_retro) - diff_retro[x_75_retro]])]
            width_retro = np.r_[width_retro,np.array(-diff_retro[x_25_retro]+ diff_retro[x_75_retro])/1.349]
    






### Dynedge 
           
axs_perf[0,1].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,1].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,1].legend()
axs_perf[0,1].grid()
axs_perf[0,1].set_title('50 %s Best Uncertainies Dynedge'%chr(37), size = 20)
axs_perf[1,1].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,1].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,1].legend()
axs_perf[1,1].grid()
axs_perf[1,1].set_xlabel('N Pulses', size = 20) 

#### Retro

# ##
# #########
# #       
# # RETRO SIGMA CUT
# #
# ########

plot_data = pd.concat( [pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[dynedge]).reshape(-1,1))*const),
                        pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[variable]).reshape(-1,1))*const),
                        pd.DataFrame(retro_mc[retro]*const),
                        pd.DataFrame(retro_mc[retro_sigma])*const,
                        pd.DataFrame(strings['n_doms'])],axis = 1)
plot_data.columns = [dynedge,variable,retro,retro_sigma,'n_doms']
plot_data.to_csv(r'X:\speciale\litterature\to_troels\mc_data_energy_log10.csv')
plot_data = plot_data.sort_values(retro)

result = plot_data.sort_values('n_doms')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['n_doms'].reset_index(drop = True)
num_bins = 25
fig3 = plt.figure()
n, bins_e, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
plt.close()
means_E = list()
medians_error = list()
medians_error_retro = list()
errors_width = list()
errors_width_retro = list()
for k in range(len(bins_e)-1):
    index = (E >= bins_e[k]) & (E <= bins_e[k+1])
    if(sum(index) != 0):
        means_E.append(np.mean(E[index]))
        medians_error.append(np.median(pred[index]-true[index]))
        medians_error_retro.append(np.median(retro_pred[index]-true[index]))
        diff = (pred - true)[index].reset_index(drop = True)
        diff_retro = (retro_pred - true)[index].reset_index(drop = True)
        x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
        x_25_retro = abs(diff_retro-np.percentile(diff_retro,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75_retro = abs(diff_retro-np.percentile(diff_retro,75,interpolation='nearest')).argmin() #int(0.84*N)
    
        N = sum(index)
        fe_25 = sum(diff <= diff[x_25])/N
        fe_75 = sum(diff <= diff[x_75])/N
        errors_width.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25**2 + 1/fe_75**2))*(1/1.349))
    
        fe_25_retro = sum(diff_retro <= diff_retro[x_25_retro])/N
        fe_75_retro = sum(diff_retro <= diff_retro[x_75_retro])/N
        errors_width_retro.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25_retro**2 + 1/fe_75_retro**2))*(1/1.349))
        if( k == 0):
            errors = np.array([np.median(diff) - diff[x_25], 
                                np.median(diff) - diff[x_75]])
            width = np.array(-diff[x_25]+ diff[x_75])/1.349
        
            errors_retro = np.array([np.median(diff_retro) - diff_retro[x_25_retro], 
                                np.median(diff_retro) - diff_retro[x_75_retro]])
            width_retro = np.array(-diff_retro[x_25_retro]+ diff_retro[x_75_retro])/1.349
        else:
            errors = np.c_[errors,np.array([np.median(diff) - diff[x_25],
                                            np.median(diff) - diff[x_75]])]
            width = np.r_[width,np.array(-diff[x_25]+ diff[x_75])/1.349]
         
            errors_retro = np.c_[errors_retro,np.array([np.median(diff_retro) - diff_retro[x_25_retro],
                                            np.median(diff_retro) - diff_retro[x_75_retro]])]
            width_retro = np.r_[width_retro,np.array(-diff_retro[x_25_retro]+ diff_retro[x_75_retro])/1.349]




fig_perf, axs_perf = plt.subplots(2, 2)
fig_perf.suptitle(' %s MC Performance: dynedge NC + CC'%variable,size = 20)

## Dynedge        
axs_perf[0,0].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,0].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,0].legend()
axs_perf[0,0].grid()
axs_perf[0,0].set_title('No Cut', size = 20)
if(variable != 'energy_log10'):
    axs_perf[0,0].set_ylabel('$\Delta ø$ [Deg.]', size = 20)
else:
    axs_perf[0,0].set_ylabel(r'$log(\frac{E_{pred}}{E}$)', size = 20)      
axs_perf[1,0].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,0].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,0].legend()
axs_perf[1,0].grid()
axs_perf[1,0].set_xlabel('N Pulses',size = 20)
if(variable != 'energy_log10'):
    axs_perf[1,0].set_ylabel('$W(\Delta ø) [Deg.]$', size = 20)
    
else:
    axs_perf[1,0].set_ylabel(r'W($log(\frac{E_{pred}}{E}$))', size = 20)
#########


########
fig, axs = plt.subplots(3, 2)

fig.suptitle(' %s MC Distributions'%variable,size = 20)

axs[0, 0].hist(plot_data[retro],bins = bins,label = 'retro')
axs[0, 0].hist(plot_data[variable],bins = bins,label = 'truth',histtype='step')
axs[0, 0].grid()

axs[0, 0].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
axs[0, 0].legend()
axs[0, 0].set_title('No Cut', size = 20)


axs[1, 0].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro')
axs[1, 0].set_ylabel('retro', size = 20)


axs[2, 0].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro')
axs[2, 0].set_xlabel('truth',size = 20)
axs[2, 0].set_ylabel('dynedge', size = 20)
axs[2, 0].legend()



print('Data Before Cut: %s'%len(plot_data))
a = len(plot_data)
n = int(len(plot_data)*k_threshold)
plot_data = plot_data.nsmallest(n,retro_sigma)

print('Data After Cut: %s'%len(plot_data))

print('Retainment: %s procent'%(len(plot_data)/a))


axs[0, 1].hist(plot_data[retro],bins = bins,label = 'retro')
axs[0, 1].hist(plot_data[variable],bins = bins,label = 'truth',histtype='step')
axs[0, 1].grid()
axs[0, 1].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
axs[0, 1].legend()
axs[0, 1].set_title('50 %s Best Uncertainties Retro'%chr(37), size = 20)
axs[1, 1].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro')
axs[2, 1].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro')
axs[2, 1].set_xlabel('truth', size = 20)
axs[2, 1].legend()




####


###### 
#
# PLOT ERROR
#
######


result = plot_data.sort_values('n_doms')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['n_doms'].reset_index(drop = True)
num_bins = 25
fig3 = plt.figure()
n, bins_e, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
plt.close()
means_E = list()
medians_error = list()
medians_error_retro = list()
errors_width = list()
errors_width_retro = list()
for k in range(len(bins_e)-1):
    index = (E >= bins_e[k]) & (E <= bins_e[k+1])
    if(sum(index) != 0):
        means_E.append(np.mean(E[index]))
        medians_error.append(np.median(pred[index]-true[index]))
        medians_error_retro.append(np.median(retro_pred[index]-true[index]))
        diff = (pred - true)[index].reset_index(drop = True)
        diff_retro = (retro_pred - true)[index].reset_index(drop = True)
        ylabel = r'$log(\frac{E_{pred}}{E}$)'
        title = r' $log(\frac{E_{pred}}{E}$) vs E'
        x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
        x_25_retro = abs(diff_retro-np.percentile(diff_retro,25,interpolation='nearest')).argmin() #int(0.16*N)
        x_75_retro = abs(diff_retro-np.percentile(diff_retro,75,interpolation='nearest')).argmin() #int(0.84*N)
    
        N = sum(index)
        fe_25 = sum(diff <= diff[x_25])/N
        fe_75 = sum(diff <= diff[x_75])/N
        errors_width.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25**2 + 1/fe_75**2))*(1/1.349))
    
        fe_25_retro = sum(diff_retro <= diff_retro[x_25_retro])/N
        fe_75_retro = sum(diff_retro <= diff_retro[x_75_retro])/N
        errors_width_retro.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25_retro**2 + 1/fe_75_retro**2))*(1/1.349))
        if( k == 0):
            errors = np.array([np.median(diff) - diff[x_25], 
                                np.median(diff) - diff[x_75]])
            width = np.array(-diff[x_25]+ diff[x_75])/1.349
        
            errors_retro = np.array([np.median(diff_retro) - diff_retro[x_25_retro], 
                                np.median(diff_retro) - diff_retro[x_75_retro]])
            width_retro = np.array(-diff_retro[x_25_retro]+ diff_retro[x_75_retro])/1.349
        else:
            errors = np.c_[errors,np.array([np.median(diff) - diff[x_25],
                                            np.median(diff) - diff[x_75]])]
            width = np.r_[width,np.array(-diff[x_25]+ diff[x_75])/1.349]
         
            errors_retro = np.c_[errors_retro,np.array([np.median(diff_retro) - diff_retro[x_25_retro],
                                            np.median(diff_retro) - diff_retro[x_75_retro]])]
            width_retro = np.r_[width_retro,np.array(-diff_retro[x_25_retro]+ diff_retro[x_75_retro])/1.349]







### Dynedge 
       
axs_perf[0,1].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,1].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,1].legend()
axs_perf[0,1].grid()
axs_perf[0,1].set_title('50 %s Best Uncertainties Retro'%chr(37), size = 20)
axs_perf[1,1].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,1].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,1].legend()
axs_perf[1,1].grid()
axs_perf[1,1].set_xlabel('N Pulses', size = 20) 


