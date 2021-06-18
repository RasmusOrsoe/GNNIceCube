import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

def MakeLabels(data):
    
    scores = data['out2']/(data['out1'] + data['out2'])
    
    out = pd.concat([pd.DataFrame(data['event_no']), pd.DataFrame(scores)], axis = 1)
    out.columns = ['event_no', 'neutrino']
    return out


db_file = r'X:\speciale\data\raw\oscnext_IC8611_newfeats_000_mc_scaler\data\oscnext_IC8611_newfeats_000_mc_scaler.db'

mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'

#scalers_retro = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_oscNext_IC86_11_with_retro\meta\transformers.pkl')

scalers_dyn = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')
pid_scaler = pd.read_pickle(r'X:\speciale\data\raw\IC8611_oscNext_003_final\meta\transformers.pkl')['truth']['lvl7_probnu']
#retro_scaler = pd.read_pickle(r'')
results = pd.read_csv(r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler\regression\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')#r'X:\speciale\results\dev_level7_oscNext_IC86_003\event_only_level7_all_oscNext_IC86_003\dynedge-E-protov2-zenith\results.csv')

mc_res = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything_wtest_testset\dynedge-protov2-zenith-k=8-c3not3-w_test_val\results.csv')#r'X:\speciale\results\dev_level7_mu_tau_e_retro_000\event_only_level7_all_neutrinos_retro_SRT_4mio\dynedge-E-protov2-zenith\results.csv')

labels = pd.read_csv(r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler\regression\dynedge-protov2-classification-k=8-c3not3-classification-even\labels.csv')
labels.columns = ['a','event_no', 'neutrino']
labels = labels.sort_values('event_no').reset_index(drop = True)#MakeLabels(pid_data).sort_values('event_no').reset_index(drop = True)
pid_retro = pd.read_csv(r'X:\speciale\data\raw\oscnext_IC8611_newfeats_000_mc_scaler\data\lvl7_probnu.csv')
pid_retro = pid_retro.sort_values('event_no').reset_index(drop = True)

results = results.sort_values('event_no').reset_index(drop = True)
events = results['event_no']
with sqlite3.connect(db_file) as con:
    query = 'select * from truth where event_no in %s'%str(tuple(events))
    data = pd.read_sql(query,con)
    query = 'select * from features where event_no in %s'%str(tuple(events))
    feats = pd.read_sql(query,con)

data = data.sort_values('event_no').reset_index(drop = True)

variable = 'zenith'
plot_osc = False
pid      = None
k_threshold = 0.50
pid_threshold = 0.7
dynedge = variable + '_pred'
retro   = variable + '_retro'
retro_sigma = variable + '_sigma'
sigma   = dynedge + '_k'
it = None
E_bins = np.arange(0,4.5,0.5)
cmap_hist=plt.get_cmap('Blues')
ycorr = 50
if variable == 'azimuth':
    const = 360/(2*np.pi)
    bins = np.arange(0,6.1,0.1)*const
    #unit = ''
if variable == 'zenith':
    const = 360/(2*np.pi)
    bins = np.arange(0,3.1,0.1)*const
if variable == 'energy_log10':
    const = 1
    bins = np.arange(0,4,0.1)

ab = data

bc = results

if variable == 'energy_log10':
    plot_data_real = pd.concat( [pd.DataFrame(results['event_no']),
                               pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(results[dynedge]).reshape(-1,1))*const), 
                               pd.DataFrame(data[retro])*const,
                               pd.DataFrame(data[retro_sigma])*const,
                               pd.DataFrame(labels['neutrino']),
                               pd.DataFrame(pid_scaler.inverse_transform(np.array(pid_retro['lvl7_probnu']).reshape(-1,1)))],axis = 1)
    
    plot_data_real.columns = ['event_no',dynedge,retro,retro_sigma, 'neutrino', 'lvl7_probnu']
    
if variable == 'zenith':
    plot_data_real = pd.concat( [pd.DataFrame(results['event_no']),
                               pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(results[dynedge]).reshape(-1,1))*const), 
                               pd.DataFrame(data[retro])*const,
                               pd.DataFrame(data[retro_sigma])*const,
                               pd.DataFrame(1/np.sqrt(abs(results[sigma]))*const),
                               pd.DataFrame(labels['neutrino']),
                               pd.DataFrame(pid_scaler.inverse_transform(np.array(pid_retro['lvl7_probnu']).reshape(-1,1)))],axis = 1)
    plot_data_real.columns = ['event_no',dynedge,retro,retro_sigma,sigma, 'neutrino', 'lvl7_probnu']
    
if variable == 'azimuth':
    plot_data_real = pd.concat( [pd.DataFrame(results['event_no']),
                               pd.DataFrame(np.array(results[dynedge]))*const, 
                               pd.DataFrame(data[retro])*const,
                               pd.DataFrame(data[retro_sigma])*const,
                               pd.DataFrame(1/np.sqrt(abs(results[sigma]))*const),
                               pd.DataFrame(labels['neutrino']),
                               pd.DataFrame(pid_scaler.inverse_transform(np.array(pid_retro['lvl7_probnu']).reshape(-1,1)))],axis = 1)


    plot_data_real.columns = ['event_no',dynedge,retro,retro_sigma,sigma, 'neutrino', 'lvl7_probnu']
    index = plot_data_real[dynedge]<0
    plot_data_real[dynedge][index] = plot_data_real[dynedge][index] + 2*np.pi*const
    index = plot_data_real[dynedge]>2*np.pi*const
    plot_data_real[dynedge][index] = plot_data_real[dynedge][index] - 2*np.pi*const

plot_data_real = plot_data_real.sample(frac = 0.1).reset_index(drop = True)
plot_data_real.to_csv(r'X:\speciale\litterature\to_troels\real_data_%s.csv'%variable)
#plot_data_real = plot_data_real.sort_values(retro)


fig, ax = plt.subplots(4, 2)
fig.suptitle('IC86.11 OscNext Real Data: %s predictions'%variable,size = 20)

neutrino_index = plot_data_real['neutrino'] > pid_threshold
retro_pid_index = plot_data_real['lvl7_probnu'] > pid_threshold

ax[0,0].hist(plot_data_real[retro],bins = bins,label = 'retro', histtype = 'step')
ax[0,0].hist(plot_data_real[dynedge],histtype='step',bins = bins,label = 'dynedge')
ax[0,0].set_title('No Cut', size = 20)

ax[0,0].legend()
ax[0,0].set_title('No Cut', size = 20)

ax[1,0].hist2d(plot_data_real[retro],plot_data_real[dynedge],bins = bins,cmap = cmap_hist)
ax[1,0].plot(plot_data_real[retro],plot_data_real[retro],'--', color = 'grey' )
ax[1,0].set_ylabel('$%s_{dynedge}$ [Deg.]'%variable,size =  10)
ax[2,0].hist2d(plot_data_real[retro][neutrino_index],plot_data_real[dynedge][neutrino_index], bins = bins ,cmap = cmap_hist)
ax[2,0].plot(plot_data_real[retro],plot_data_real[retro],'--', color = 'grey' )
ax[2,0].set_ylabel('$%s_{dynedge}$ [Deg.]'%variable, size = 10)

#ax[3,0].hist2d(plot_data_real[retro][retro_pid_index],plot_data_real[dynedge][retro_pid_index], bins = bins,cmap = cmap_hist)
#ax[3,0].set_ylabel('$%s_{dynedge}$ [Deg.]'%variable, size = 15)

#ax[3,0].set_ylabel('$dynedge_{\sigma}$ [Deg.]', size = 20)

p_x = plot_data_real[retro].mean()
p_y = plot_data_real[dynedge].max() - 2/5*plot_data_real[dynedge].max()
ax[1,0].text(p_x,p_y- ycorr, '10% of IC86.11 OscNext', fontsize=15,  color='red')


fig_sigma, ax_sigma = plt.subplots(2, 2)
fig_sigma.suptitle('IC86.11 OscNext Real Data: %s $\sigma$'%variable,size = 20)

if variable != 'energy_log10':
    not_nan = ~np.isnan(plot_data_real[sigma])
    ax_sigma[0,0].hist(plot_data_real[retro_sigma][not_nan], label =  'retro', histtype = 'step', bins = np.arange(0,plot_data_real[retro_sigma].max(),0.2))
    ax_sigma[0,0].hist(plot_data_real[sigma][not_nan], label =  'dynedge', histtype = 'step',  bins = np.arange(0,plot_data_real[retro_sigma].max(),0.2))
    ax_sigma[0,0].set_title('No Cut', size = 20)
    ax_sigma[0,0].set_xlabel('$\sigma$ [Deg.]', size = 20)
    ax_sigma[1,0].hist2d(plot_data_real[retro_sigma][not_nan],plot_data_real[sigma][not_nan], bins = np.arange(0,50,1))
    ax_sigma[1,0].set_xlabel('$retro_{\sigma}$ [Deg.]', size = 20)
    ax_sigma[1,0].set_ylabel('$dynedge_{\sigma}$ [Deg.]', size = 20)
    ax_sigma[0,0].set_title('No Cut', size = 20)
    ax_sigma[0,0].legend()

    




####
#
# Real K cut
#
#####

n = int(len(plot_data_real)*k_threshold)
if variable != 'energy_log10':
    plot_data_real = plot_data_real.nsmallest(n, sigma).sort_values('event_no')
    neutrino_index = plot_data_real['neutrino'] > pid_threshold
    retro_pid_index = plot_data_real['lvl7_probnu'] > pid_threshold
    
    ax[0,1].hist(plot_data_real[retro],bins = bins,label = 'retro', histtype = 'step')
    ax[0,1].hist(plot_data_real[dynedge],histtype='step',bins = bins,label = 'dynedge')
    ax[0,1].set_title('%s %s Most Certain Dynedge'%(k_threshold*100,chr(37)), size = 20)
    ax[2,1].hist2d(plot_data_real[retro][neutrino_index],plot_data_real[dynedge][neutrino_index], bins = bins,cmap = cmap_hist)
    ax[2,1].plot(plot_data_real[retro],plot_data_real[retro],'--', color = 'grey' )
    ax[2,1].text(p_x,p_y- ycorr, '$\\nu$-classified 10% IC86.11 OscNext', fontsize=15,  color='red')
    #ax[3,1].hist2d(plot_data_real[retro][retro_pid_index],plot_data_real[dynedge][retro_pid_index], bins = bins,cmap = cmap_hist)
    #ax[3,1].text(p_x,p_y, '$\\nu$-classified retro IC86.11 OscNext', fontsize=15,  color='red')
    #ax[3,1].set_xlabel('$retro_{\sigma}$ [Deg.]', size = 20)
    #ax[2,1].set_ylabel('dynedge', size = 20)
    ax[0,1].legend()
    ax[0,1].set_title('%s %s Most Certain Dynedge'%(k_threshold*100,chr(37)), size = 20)
    
    ax[1,1].hist2d(plot_data_real[retro],plot_data_real[dynedge],bins = bins,cmap = cmap_hist)
    ax[1,1].plot(plot_data_real[retro],plot_data_real[retro],'--', color = 'grey' )
    ax[1,1].text(p_x,p_y - ycorr, '10% IC86.11 OscNext', fontsize=15,  color='red')
    
    not_nan = ~np.isnan(plot_data_real[sigma])
    ax_sigma[0,1].hist(plot_data_real[retro_sigma][not_nan], label =  'retro', histtype = 'step', bins = np.arange(0,plot_data_real[retro_sigma].max(),0.2))
    ax_sigma[0,1].hist(plot_data_real[sigma][not_nan], label =  'dynedge', histtype = 'step',  bins = np.arange(0,plot_data_real[retro_sigma].max(),0.2))
    ax_sigma[0,1].set_xlabel('$\sigma$ [Deg.]', size = 20)
    ax_sigma[0,1].set_title('%s %s Most Certain Dynedge'%(k_threshold*100,chr(37)), size = 20)

    ax_sigma[1,1].hist2d(plot_data_real[retro_sigma][not_nan],plot_data_real[sigma][not_nan], bins = np.arange(0,50,1),cmap = cmap_hist)
    ax_sigma[1,1].set_xlabel('$retro_{\sigma}$ [Deg.]', size = 20)
    #ax_sigma[1,1].set_ylabel('$dynedge_{\sigma}$ [Deg.]', size = 20)
    ax_sigma[0,1].legend()


##### MC
mc_res =  mc_res.sort_values('event_no').reset_index(drop = True)
mc_events = mc_res['event_no']
#r'X:\speciale\data\raw\standard_reco\reco\retro_reco_2.db'
with sqlite3.connect(mc_db) as con:
    query = 'select * from truth where event_no in %s'%str(tuple(mc_events))
    retro_mc = pd.read_sql(query,con)
    

retro_mc = retro_mc.sort_values('event_no').reset_index(drop = True)

if variable == 'azimuth':
    index = mc_res[dynedge]<0
    mc_res[dynedge][index] = mc_res[dynedge][index] + 2*np.pi
    index = mc_res[dynedge]>2*np.pi
    mc_res[dynedge][index] = mc_res[dynedge][index] - 2*np.pi
    plot_data = pd.concat( [pd.DataFrame(mc_res['event_no']),
                        pd.DataFrame(np.array(mc_res[dynedge]).reshape(-1,1)*const),
                        pd.DataFrame(np.array(mc_res[variable])*const),
                        pd.DataFrame(retro_mc[retro]*const),
                        pd.DataFrame(1/np.sqrt(abs(mc_res[sigma]))*const),
                        pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(retro_mc['energy_log10']).reshape(-1,1))),
                        pd.DataFrame(retro_mc['pid']),
                        pd.DataFrame(retro_mc['interaction_type'])],axis = 1)
    plot_data.columns = ['event_no',dynedge,variable,retro,sigma,'energy_log10','pid','interaction_type']

if variable == 'energy_log10':
    plot_data = pd.concat( [pd.DataFrame(mc_res['event_no']),
                            pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[dynedge]).reshape(-1,1))*const),
                            pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[variable]).reshape(-1,1))*const),
                            pd.DataFrame(retro_mc[retro]*const),
                            pd.DataFrame(retro_mc['pid']),
                            pd.DataFrame(np.array(retro_mc['osc_weight'])),
                            pd.DataFrame(retro_mc['interaction_type'])],axis = 1)
    plot_data.columns = ['event_no',dynedge,variable,retro,'pid','osc_weight','interaction_type']
if variable == 'zenith':
    plot_data = pd.concat( [pd.DataFrame(mc_res['event_no']),
                            pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[dynedge]).reshape(-1,1))*const),
                            pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[variable]).reshape(-1,1))*const),
                            pd.DataFrame(retro_mc[retro]*const),
                            pd.DataFrame(1/np.sqrt(abs(mc_res[sigma]))*const),
                            pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(retro_mc['energy_log10']).reshape(-1,1))),
                            pd.DataFrame(retro_mc['pid']),
                            pd.DataFrame(np.array(retro_mc['osc_weight'])),
                            pd.DataFrame(retro_mc['interaction_type'])],axis = 1)
    plot_data.columns = ['event_no',dynedge,variable,retro,sigma,'energy_log10','pid','osc_weight','interaction_type']


        


plot_data = plot_data.reset_index(drop = True)


print('Data before PID cut: %s'%len(plot_data))
if pid != None:
    plot_data = plot_data.loc[abs(plot_data['pid']) == pid,:]
    if pid == 14:
        pid_tag = ' $v_{\mu}$'
    if pid == 12:
        pid_tag = ' $v_{e}$'
    if pid == 16:
        pid_tag = ' $v_{\\tau}$'
else:
    pid_tag  = ' ($v_{\mu, e, \\tau}$)'
    
if it == 1:
    pid_tag = pid_tag + 'CC'
elif it == 2:
    pid_tag = pid_tag + 'NC'
else:
    pid_tag = pid_tag +' CC + NC'
print('Data after PID cut: %s'%len(plot_data))
if it != None:
    plot_data = plot_data.loc[abs(plot_data['interaction_type']) == it,:]
    plot_data = plot_data.reset_index(drop = True)

print('Data after IT cut: %s'%len(plot_data))


if plot_osc:
    fig_osc_mc, ax_osc = plt.subplots(3, 2)
    cmap=plt.get_cmap('hot')
    fig_osc_mc.suptitle(' %s MC Oscillation'%variable  + pid_tag,size = 20)
    energy = plot_data['energy_log10']
    #zenith = scalers['zenith'].inverse_transform(np.array(data_osc['zenith']).reshape(-1,1))
    cos_dyn = np.cos(plot_data[dynedge]/const)#np.cos(plot_data[dynedge]/const)
    cos_retro = np.cos(plot_data[retro]/const)
    cos = np.cos(plot_data[variable]/const)
    weights = plot_data['osc_weight']
    ax_osc[0,0].hist2d(energy.squeeze(),cos_dyn.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)
    ax_osc[1,0].hist2d(energy.squeeze(),cos_retro.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)
    
    ax_osc[2,0].hist2d(energy.squeeze(),cos.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)
    
    #plt.xlabel("$Energy_{log10}$",size = 20)
    ax_osc[0,0].set_title('No Cut')
    ax_osc[2,0].set_ylabel("$cos(ø)_{true}$",size = 20)
    ax_osc[0,0].set_ylabel("$cos(ø)_{dynedge}$",size = 20)
    ax_osc[1,0].set_ylabel("$cos(ø)_{retro}$",size = 20)
    ax_osc[2,0].set_xlabel('$Log_{10}(E_{True})$ [GeV]', size = 20)
    #plt.colorbar()


if variable != 'energy_log10':
    ax[3,0].hist2d(plot_data[retro],plot_data[dynedge],bins = bins,cmap = cmap_hist)
    ax[3,0].plot(plot_data_real[retro],plot_data_real[retro],'--', color = 'grey' )
    ax[3,0].set_ylabel('$%s_{dynedge}$ [Deg.]'%variable,size =  10)
    ax[3,0].set_xlabel('$%s_{retro}$ [Deg.]'%variable,size =  20)
    p_x = plot_data[retro].mean()
    p_y = plot_data[dynedge].max() - 2/5*plot_data[dynedge].max()
    ax[3,0].text(p_x,p_y - ycorr, 'MC' + pid_tag, fontsize=20,  color='red')
    ax[2,0].text(p_x,p_y- ycorr, '$\\nu$-classified 10% IC86.11 OscNext', fontsize=15,  color='red')


###### 
#
# PLOT ERROR
#
######


result = plot_data.sort_values('energy_log10')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['energy_log10'].reset_index(drop = True)
num_bins = E_bins
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
        if variable == 'azimuth':
            diff[diff > np.pi*const] = diff[diff > np.pi*const] -np.pi*const
            diff[diff < -np.pi*const] = diff[diff < -np.pi*const] +np.pi*const
            diff_retro[diff_retro > np.pi*const] = diff_retro[diff_retro > np.pi*const] -np.pi*const
            diff_retro[diff_retro < -np.pi*const] = diff_retro[diff_retro < -np.pi*const] +np.pi*const
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
fig_perf.suptitle(' %s MC Performance'%variable  + pid_tag,size = 20)

dynedge_col = []
retro_col = []
dynedge_col.append([means_E,medians_error, width])
retro_col.append([means_E, medians_error_retro, width_retro])
## Dynedge        
axs_perf[0,0].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,0].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,0].legend()
axs_perf[0,0].grid()
axs_perf[0,0].set_title('No Cut', size = 20)
if(variable != 'energy_log10'):
    axs_perf[0,0].set_ylabel('$\Delta ø$ [Deg.]', size = 20)
else:
    axs_perf[0,0].set_ylabel('$log_{10}(\\frac{E_{pred}}{E_{True}}$)', size = 20)
          
axs_perf[1,0].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,0].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,0].legend()
axs_perf[1,0].grid()
if(variable != 'energy_log10'):
    axs_perf[1,0].set_ylabel('$W(\Delta ø) [Deg.]$', size = 20)
    axs_perf[1,0].set_xlabel('$Energy_{log_{10}} GeV$',size = 20)
else:
    axs_perf[1,0].set_ylabel('$log_{10}(\\frac{E_{pred}}{E_{True}}$)', size = 20)
    axs_perf[1,0].set_xlabel('$Energy_{log_{10}} GeV$',size = 20)
#########



########
fig, axs = plt.subplots(3, 2)

fig.suptitle(' %s MC Distributions'%variable  + pid_tag,size = 20)

axs[0, 0].hist(plot_data[variable],bins = bins,label = 'truth',color = 'lightgrey')
axs[0, 0].hist(plot_data[retro],bins = bins,label = 'retro', histtype = 'step')

axs[0, 0].grid()

axs[0, 0].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
axs[0, 0].legend()
axs[0, 0].set_title('No Cut', size = 20)


axs[1, 0].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro',cmap = cmap_hist)
axs[1, 0].plot(plot_data[variable],plot_data[variable],'--', color = 'grey')
axs[1, 0].set_ylabel('$%s_{retro}$ [Deg.]'%variable, size = 20)


axs[2, 0].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro',cmap = cmap_hist)
axs[2, 0].plot(plot_data[variable],plot_data[variable],'--', color = 'grey' )
axs[2, 0].set_xlabel('$%s_{truth}$'%variable,size = 20)
axs[2, 0].set_ylabel('$%s_{dynedge}$ [Deg.]'%variable, size = 20)
axs[2, 0].legend()



######
#
# K - CUT
#
######
print('Data Before Cut: %s'%len(plot_data))
a = len(plot_data)

n = int(len(plot_data)*k_threshold)
if variable != 'energy_log10':
    plot_data = plot_data.nsmallest(n, sigma).sort_values('event_no')
    for_later = plot_data
    
    print('Data After Cut: %s'%len(plot_data))
    
    print('Retainment: %s procent'%(len(plot_data)/a))
    
    axs[0, 1].hist(plot_data[variable],bins = bins,label = 'truth',color = 'lightgrey')
    axs[0, 1].hist(plot_data[retro],bins = bins,label = 'retro', histtype = 'step')
    
    axs[0, 1].grid()
    axs[0, 1].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
    axs[0, 1].legend()
    axs[0, 1].set_title('%s %s Most Certain Dynedge'%(k_threshold*100,chr(37)), size = 20)
    axs[1, 1].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro',cmap = cmap_hist)
    axs[1, 1].plot(plot_data[variable],plot_data[variable],'--', color = 'grey')
    axs[2, 1].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro',cmap = cmap_hist)
    axs[2, 1].plot(plot_data[variable],plot_data[variable],'--', color = 'grey')
    axs[2, 1].set_xlabel('$%s_{truth}$'%variable, size = 20)
    axs[2, 1].legend()



### osc
if plot_osc:
    energy = plot_data['energy_log10']
    #zenith = scalers['zenith'].inverse_transform(np.array(data_osc['zenith']).reshape(-1,1))
    cos_dyn = np.cos(plot_data[dynedge]/const)#np.cos(plot_data[dynedge]/const)
    cos_retro = np.cos(plot_data[retro]/const)
    cos = np.cos(plot_data[variable]/const)
    weights = plot_data['osc_weight']
    ax_osc[0,1].hist2d(energy.squeeze(),cos_dyn.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)
    ax_osc[1,1].hist2d(energy.squeeze(),cos_retro.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)
    
    ax_osc[2,1].hist2d(energy.squeeze(),cos.squeeze(), weights = weights.squeeze(), bins = 100,cmap = cmap)
    ax_osc[2,1].set_xlabel('$Log_{10}(E_{True})$ [GeV]', size = 20)
    #plt.xlabel("$Energy_{log10}$",size = 20)
    ax_osc[0,1].set_title('%s %s Most Certain Dynedge'%(k_threshold*100,chr(37)), size = 20)
    #ax_osc[2,1].set_ylabel("$cos(ø)_{true}$",size = 20)
    #ax_osc[0,1].set_ylabel("$cos(ø)_{dynedge}$",size = 20)
    #ax_osc[1,1].set_ylabel("$cos(ø)_{retro}$",size = 20)



#### This is the very first plot with real data in it

ax[3,1].hist2d(plot_data[retro],plot_data[dynedge],bins = bins,cmap = cmap_hist)
ax[3,1].plot(plot_data_real[retro],plot_data_real[retro],'--', color = 'grey' )
#ax[2,1].set_ylabel('$predictions_{dynedge}$',size =  20)
ax[3,1].set_xlabel('$%s_{retro}$ [Deg.]'%variable,size =  20)
ax[3,1].text(p_x,p_y- ycorr, 'MC'  + pid_tag, fontsize=20,  color='red')

####


###### 
#
# PLOT ERROR
#
######


result = plot_data.sort_values('energy_log10')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['energy_log10'].reset_index(drop = True)
num_bins = E_bins
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
        if variable == 'azimuth':
            diff[diff > np.pi*const] = diff[diff > np.pi*const] -np.pi*const
            diff[diff < -np.pi*const] = diff[diff < -np.pi*const] +np.pi*const
            diff_retro[diff_retro > np.pi*const] = diff_retro[diff_retro > np.pi*const] -np.pi*const
            diff_retro[diff_retro < -np.pi*const] = diff_retro[diff_retro < -np.pi*const] +np.pi*const
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
    





dynedge_col.append([means_E,medians_error, width])
retro_col.append([means_E, medians_error_retro, width_retro])
### Dynedge 
           
axs_perf[0,1].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,1].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,1].legend()
axs_perf[0,1].grid()
axs_perf[0,1].set_title('%s %s Most Certain Dynedge'%(k_threshold*100,chr(37)), size = 20)
axs_perf[1,1].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,1].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,1].legend()
axs_perf[1,1].grid()
axs_perf[1,1].set_xlabel('$Energy_{log_{10}} GeV$', size = 20) 

#### Retro

##
#########
#       
# RETRO SIGMA CUT
#
########
if variable == 'azimuth':
    index = mc_res[dynedge]<0
    mc_res[dynedge][index] = mc_res[dynedge][index] + 2*np.pi
    index = mc_res[dynedge]>2*np.pi
    mc_res[dynedge][index] = mc_res[dynedge][index] - 2*np.pi
    plot_data = pd.concat( [pd.DataFrame(mc_res[dynedge]*const),
                        pd.DataFrame(np.array(mc_res[variable])*const),
                        pd.DataFrame(retro_mc[retro]*const),
                        pd.DataFrame(retro_mc[retro_sigma]*const),
                        pd.DataFrame(1/np.sqrt(mc_res[sigma])*const),
                        pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(mc_res['energy_log10']).reshape(-1,1))),
                        pd.DataFrame(retro_mc['pid']),
                        pd.DataFrame(retro_mc['interaction_type'])],axis = 1)
    plot_data.columns = [dynedge,variable,retro,retro_sigma,sigma,'energy_log10','pid','interaction_type']

if variable == 'energy_log10':
    plot_data = pd.concat( [pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[dynedge]).reshape(-1,1))*const),
                            pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[variable]).reshape(-1,1))*const),
                            pd.DataFrame(retro_mc[retro]*const),
                            pd.DataFrame(retro_mc[retro_sigma]*const),
                            pd.DataFrame(retro_mc['pid']),
                            pd.DataFrame(retro_mc['interaction_type'])],axis = 1)

    plot_data.columns = [dynedge,variable,retro,retro_sigma,'pid','interaction_type']
    
if variable == 'zenith':   
    plot_data = pd.concat( [pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[dynedge]).reshape(-1,1))*const),
                            pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(mc_res[variable]).reshape(-1,1))*const),
                            pd.DataFrame(retro_mc[retro]*const),
                            pd.DataFrame(retro_mc[retro_sigma]*const),
                            pd.DataFrame(1/np.sqrt(mc_res[sigma])*const),
                            pd.DataFrame(scalers_dyn['truth']['energy_log10'].inverse_transform(np.array(mc_res['energy_log10']).reshape(-1,1))),
                            pd.DataFrame(retro_mc['pid']),
                            pd.DataFrame(retro_mc['interaction_type'])],axis = 1)

    plot_data.columns = [dynedge,variable,retro,retro_sigma,sigma,'energy_log10','pid','interaction_type']
plot_data.to_csv(r'X:\speciale\litterature\to_troels\mc_data_%s.csv'%variable)

if it == 1:
    it_tag = 'CC'
elif it == 2:
    it_tag = 'NC'
else:
    it_tag = 'CC + NC'

if pid != None:
    plot_data = plot_data.loc[abs(plot_data['pid']) == pid,:]
    if pid == 14:
        pid_tag = ' $v_{\mu,%s}$'%it_tag
    if pid == 12:
        pid_tag = ' $v_{e,%s}$'%it_tag
    if pid == 16:
        pid_tag = ' $v_{\\tau,%s}$'%it_tag
else:
    pid_tag  = ' ($v_{\mu, e, \\tau,%s}$)'%it_tag

print('Data after PID cut: %s'%len(plot_data))
if it != None:
    plot_data = plot_data.loc[abs(plot_data['interaction_type']) == it,:]



result = plot_data.sort_values('energy_log10')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['energy_log10'].reset_index(drop = True)
num_bins = E_bins
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
        if variable == 'azimuth':
            diff[diff > np.pi*const] = diff[diff > np.pi*const] -np.pi*const
            diff[diff < -np.pi*const] = diff[diff < -np.pi*const] +np.pi*const
            diff_retro[diff_retro > np.pi*const] = diff_retro[diff_retro > np.pi*const] -np.pi*const
            diff_retro[diff_retro < -np.pi*const] = diff_retro[diff_retro < -np.pi*const] +np.pi*const
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
    



fig_perf, axs_perf = plt.subplots(2, 2,figsize = (8.3 ,11.7))
fig_perf.suptitle(' %s MC Performance'%variable  + pid_tag,size = 40)

## Dynedge        
axs_perf[0,0].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,0].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,0].legend()
axs_perf[0,0].grid()
axs_perf[0,0].set_title('No Cut', size = 20)
if(variable != 'energy_log10'):
    axs_perf[0,0].set_ylabel('$\Delta ø$ [Deg.]', size = 20)
else:
    axs_perf[0,0].set_ylabel('$log_{10}(\\frac{E_{pred}}{E_{true}})$ [GeV]', size = 30)
          
axs_perf[1,0].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,0].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,0].legend()
axs_perf[1,0].grid()
if(variable != 'energy_log10'):
    axs_perf[1,0].set_ylabel('$W(\Delta ø) [Deg.]$', size = 20)
    axs_perf[1,0].set_xlabel('$Energy_{log_{10}} GeV$',size = 20)
    
else:
    axs_perf[1,0].set_ylabel('$W(log_{10}(\\frac{E_{pred}}{E_{true}})) [GeV]$', size = 30)
    axs_perf[1,0].set_xlabel('$Energy_{log_{10}} [GeV]$',size = 30)
    axs_perf[1,1].set_xlabel('$Energy_{log_{10}} [GeV]$',size = 30)
    axs_perf[0,0].tick_params(axis='x', labelsize=20)
    axs_perf[0,0].tick_params(axis='y', labelsize=20)
    axs_perf[1,0].tick_params(axis='x', labelsize=20)
    axs_perf[1,0].tick_params(axis='y', labelsize=20)
    axs_perf[0,1].tick_params(axis='x', labelsize=20)
    axs_perf[0,1].tick_params(axis='y', labelsize=20)
    axs_perf[1,1].tick_params(axis='x', labelsize=20)
    axs_perf[1,1].tick_params(axis='y', labelsize=20)
#########


########
fig, axs = plt.subplots(3, 2)

fig.suptitle(' %s MC Distributions'%variable  + pid_tag,size = 20)


axs[0, 0].hist(plot_data[variable],bins = bins,label = 'truth',color = 'lightgrey')
axs[0, 0].hist(plot_data[retro],bins = bins,label = 'retro', histtype = 'step')
axs[0, 0].grid()

axs[0, 0].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
axs[0, 0].legend()
axs[0, 0].set_title('No Cut', size = 20)


axs[1, 0].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro',cmap = cmap_hist)
axs[1, 0].plot(plot_data[variable],plot_data[variable],'--', color = 'grey')
axs[1, 0].set_ylabel('$%s_{retro}$ [Deg.]'%variable, size = 20)


axs[2, 0].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro',cmap = cmap_hist)
axs[2, 0].plot(plot_data[variable],plot_data[variable],'--', color = 'grey')
axs[2, 0].set_xlabel('$%s_{truth}$'%variable,size = 20)
axs[2, 0].set_ylabel('$%s_{dynedge}$ [Deg.]'%variable, size = 20)
axs[2, 0].legend()



print('Data Before Cut: %s'%len(plot_data))
a = len(plot_data)

n = int(len(plot_data)*k_threshold)

plot_data = plot_data.nsmallest(n, retro_sigma)

print('Data After Cut: %s'%len(plot_data))

print('Retainment: %s procent'%(len(plot_data)/a))

label_size = 15
axs[0, 1].hist(plot_data[variable],bins = bins,label = 'truth',color = 'lightgrey')
axs[0, 1].hist(plot_data[retro],bins = bins,label = 'retro', histtype = 'step')
axs[0, 1].grid()
axs[0, 1].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
axs[0, 1].legend()
axs[0, 1].set_title('%s %s Most Certain Retro'%(k_threshold*100,chr(37)), size = 20)
axs[1, 1].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro',cmap = cmap_hist)
axs[1, 1].plot(plot_data[variable],plot_data[variable],'--', color = 'grey')
axs[2, 1].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro',cmap = cmap_hist)
axs[2, 1].plot(plot_data[variable],plot_data[variable],'--', color = 'grey')
axs[2, 1].set_xlabel('$%s_{truth}$'%variable, size = 20)
axs[2, 1].legend()
axs[0,0].tick_params(axis='x', labelsize=label_size)
axs[0,0].tick_params(axis='y', labelsize=label_size)
axs[1,0].tick_params(axis='x', labelsize=label_size)
axs[1,0].tick_params(axis='y', labelsize=label_size)
axs[2,1].tick_params(axis='y', labelsize=label_size)
axs[2,1].tick_params(axis='x', labelsize=label_size)
axs[0,1].tick_params(axis='x', labelsize=label_size)
axs[0,1].tick_params(axis='y', labelsize=label_size)
axs[1,1].tick_params(axis='x', labelsize=label_size)
axs[1,1].tick_params(axis='y', labelsize=label_size)
axs[2,0].tick_params(axis='y', labelsize=label_size)
axs[2,0].tick_params(axis='x', labelsize=label_size)

####


###### 
#
# PLOT ERROR
#
######


result = plot_data.sort_values('energy_log10')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['energy_log10'].reset_index(drop = True)
num_bins = E_bins
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
        if variable == 'azimuth':
            diff[diff > np.pi*const] = diff[diff > np.pi*const] -np.pi*const
            diff[diff < -np.pi*const] = diff[diff < -np.pi*const] +np.pi*const
            diff_retro[diff_retro > np.pi*const] = diff_retro[diff_retro > np.pi*const] -np.pi*const
            diff_retro[diff_retro < -np.pi*const] = diff_retro[diff_retro < -np.pi*const] +np.pi*const
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
axs_perf[0,1].set_title('%s %s Most Certain Retro'%(k_threshold*100,chr(37)), size = 20)
axs_perf[1,1].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,1].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,1].legend()
axs_perf[1,1].grid()
axs_perf[1,1].set_xlabel('$Energy_{log_{10}} GeV$', size = 20) 


rel_imp = 1- np.array(pd.DataFrame(dynedge_col).loc[0,2])/(np.array(pd.DataFrame(retro_col).loc[0,2]))
rel_imp2 = 1 -np.array(pd.DataFrame(dynedge_col).loc[1,2])/(np.array(pd.DataFrame(retro_col).loc[1,2]))
binning = np.array(pd.DataFrame(dynedge_col).loc[0,0])

path = r'X:\speciale\litterature\conclusion\%s.csv'%variable
out = pd.DataFrame()
out['E'] = binning
out['precut'] = rel_imp
out['cut'] = rel_imp2
out.to_csv(path)