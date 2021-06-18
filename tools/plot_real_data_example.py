import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np



plot_data_real = pd.read_csv(r'X:\speciale\litterature\to_troels\real_data_zenith.csv')
plot_data = pd.read_csv(r'X:\speciale\litterature\to_troels\mc_data_zenith.csv')

plot_data = plot_data.reset_index(drop = True)
plot_data_real = plot_data_real.reset_index(drop = True)

db_file = r'X:\speciale\data\raw\dev_level7_oscNext_IC86_11_with_retro\data\dev_level7_oscNext_IC86_11_with_retro.db'



variable = 'zenith'
pid      = 14
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
if variable == 'energy_log10':
    const = 1
    bins = np.arange(0,4,0.1)


fig, ax = plt.subplots(3, 2)
fig.suptitle('IC86.11 OscNext Real Data: %s predictions'%variable,size = 20)

ax[0,0].hist(plot_data_real[retro],bins = bins,label = 'retro')
ax[0,0].hist(plot_data_real[dynedge],histtype='step',bins = bins,label = 'dynedge')
ax[0,0].set_title('No Cut', size = 20)

ax[0,0].legend()
ax[0,0].set_title('No Cut', size = 20)

ax[1,0].hist2d(plot_data_real[retro],plot_data_real[dynedge],bins = bins)
ax[1,0].set_ylabel('$predictions_{dynedge}$',size =  20)


p_x = plot_data_real[retro].mean()
p_y = plot_data_real[dynedge].max() - 2/5*plot_data_real[dynedge].max()
ax[1,0].text(p_x,p_y, 'IC86.11 OscNext', fontsize=15,  color='red')


fig_sigma, ax_sigma = plt.subplots(1, 2)
fig_sigma.suptitle('IC86.11 OscNext Real Data: %s $\sigma$'%variable,size = 20)
not_nan = ~np.isnan(plot_data_real[sigma])
ax_sigma[0].hist(plot_data_real[retro_sigma][not_nan], label =  'retro', histtype = 'step', bins = np.arange(0,plot_data_real[retro_sigma].max(),0.2))
ax_sigma[0].hist(plot_data_real[sigma][not_nan], label =  'dynedge', histtype = 'step',  bins = np.arange(0,plot_data_real[retro_sigma].max(),0.2))
ax_sigma[0].set_title('No Cut', size = 20)
ax_sigma[0].set_xlabel('$\sigma$ [Deg.]', size = 20)
ax_sigma[0].legend()




####
#
# Real K cut
#
#####

plot_data_real = plot_data_real.loc[plot_data_real[sigma]< k_threshold*plot_data_real[sigma].mean(),:]

ax[0,1].hist(plot_data_real[retro],bins = bins,label = 'retro')
ax[0,1].hist(plot_data_real[dynedge],histtype='step',bins = bins,label = 'dynedge')
ax[0,1].set_title('$K < %s \cdot k_{mean}$'%k_threshold, size = 20)

ax[0,1].legend()
ax[0,1].set_title('$K < %s \cdot k_{mean}$'%k_threshold, size = 20)

ax[1,1].hist2d(plot_data_real[retro],plot_data_real[dynedge],bins = bins)
ax[1,1].text(p_x,p_y, 'IC86.11 OscNext', fontsize=15,  color='red')

not_nan = ~np.isnan(plot_data_real[sigma])
ax_sigma[1].hist(plot_data_real[retro_sigma][not_nan], label =  'retro', histtype = 'step', bins = np.arange(0,plot_data_real[retro_sigma].max(),0.2))
ax_sigma[1].hist(plot_data_real[sigma][not_nan], label =  'dynedge', histtype = 'step',  bins = np.arange(0,plot_data_real[retro_sigma].max(),0.2))
ax_sigma[1].set_xlabel('$\sigma$ [Deg.]', size = 20)
ax_sigma[1].set_title('$K < %s \cdot k_{mean}$'%k_threshold, size = 20)
ax_sigma[1].legend()


##### MC

print('Data before PID cut: %s'%len(plot_data))
if pid != None:
    plot_data = plot_data.loc[abs(plot_data['pid']) == pid,:]
    if pid == 14:
        pid_tag = ' $v_{\mu}$'
    if pid == 12:
        pid_tag = ' $v_{e}$'
    if pid == 16:
        pid_tag = ' $v_{\tau}$'
else:
    pid_tag  = ' ($v_{\mu, e, \tau}$)'
print('Data after PID cut: %s'%len(plot_data))



ax[2,0].hist2d(plot_data[retro],plot_data[dynedge],bins = bins)
ax[2,0].set_ylabel('$predictions_{dynedge}$',size =  20)
ax[2,0].set_xlabel('$predictions_{retro}$',size =  20)
p_x = plot_data[retro].mean()
p_y = plot_data[dynedge].max() - 2/5*plot_data[dynedge].max()
ax[2,0].text(p_x,p_y, 'MC' + pid_tag, fontsize=20,  color='red')


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
num_bins = 10
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
fig_perf.suptitle(' %s MC Performance'%variable  + pid_tag,size = 20)


## Dynedge        
axs_perf[0,0].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,0].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,0].legend()
axs_perf[0,0].grid()
axs_perf[0,0].set_title('No Cut', size = 20)
if(variable != 'energy_log10'):
    axs_perf[0,0].set_ylabel('$\Delta ø$ [Deg.]', size = 20)
          
axs_perf[1,0].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,0].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,0].legend()
axs_perf[1,0].grid()
if(variable != 'energy_log10'):
    axs_perf[1,0].set_ylabel('$W(\Delta ø) [Deg.]$', size = 20)
    axs_perf[1,0].set_xlabel('$Energy_{log_{10}} GeV$',size = 20)
#########



########
fig, axs = plt.subplots(3, 2)

fig.suptitle(' %s MC Distributions'%variable  + pid_tag,size = 20)

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

plot_data = plot_data.loc[plot_data[sigma]< k_threshold*plot_data[sigma].mean(),:]

print('Data After Cut: %s'%len(plot_data))

print('Retainment: %s procent'%(len(plot_data)/a))


axs[0, 1].hist(plot_data[retro],bins = bins,label = 'retro')
axs[0, 1].hist(plot_data[variable],bins = bins,label = 'truth',histtype='step')
axs[0, 1].grid()
axs[0, 1].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
axs[0, 1].legend()
axs[0, 1].set_title('$K < %s \cdot k_{mean}$'%k_threshold, size = 20)
axs[1, 1].hist2d(plot_data[variable],plot_data[retro],bins = bins,label = 'retro')
axs[2, 1].hist2d(plot_data[variable],plot_data[dynedge],bins = bins,label = 'retro')
axs[2, 1].set_xlabel('truth', size = 20)
axs[2, 1].legend()


#### This is the very first plot with real data in it

ax[2,1].hist2d(plot_data[retro],plot_data[dynedge],bins = bins)
#ax[2,1].set_ylabel('$predictions_{dynedge}$',size =  20)
ax[2,1].set_xlabel('$predictions_{retro}$',size =  20)
ax[2,1].text(p_x,p_y, 'MC'  + pid_tag, fontsize=20,  color='red')

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
num_bins = 10
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
axs_perf[0,1].set_title('$K < %s \cdot k_{mean}$'%k_threshold, size = 20)
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


result = plot_data.sort_values('energy_log10')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['energy_log10'].reset_index(drop = True)
num_bins = 10
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
fig_perf.suptitle(' %s MC Performance'%variable  + pid_tag,size = 20)

## Dynedge        
axs_perf[0,0].errorbar(means_E,medians_error,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
axs_perf[0,0].errorbar(means_E,medians_error_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
axs_perf[0,0].legend()
axs_perf[0,0].grid()
axs_perf[0,0].set_title('No Cut', size = 20)
if(variable != 'energy_log10'):
    axs_perf[0,0].set_ylabel('$\Delta ø$ [Deg.]', size = 20)
          
axs_perf[1,0].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,0].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,0].legend()
axs_perf[1,0].grid()
if(variable != 'energy_log10'):
    axs_perf[1,0].set_ylabel('$W(\Delta ø) [Deg.]$', size = 20)
    axs_perf[1,0].set_xlabel('$Energy_{log_{10}} GeV$',size = 20)
#########


########
fig, axs = plt.subplots(3, 2)

fig.suptitle(' %s MC Distributions'%variable  + pid_tag,size = 20)

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

plot_data = plot_data.loc[plot_data[retro_sigma]< k_threshold*plot_data[retro_sigma].mean(),:]

print('Data After Cut: %s'%len(plot_data))

print('Retainment: %s procent'%(len(plot_data)/a))


axs[0, 1].hist(plot_data[retro],bins = bins,label = 'retro')
axs[0, 1].hist(plot_data[variable],bins = bins,label = 'truth',histtype='step')
axs[0, 1].grid()
axs[0, 1].hist(plot_data[dynedge],histtype='step',bins = bins,label = 'dynedge')
axs[0, 1].legend()
axs[0, 1].set_title('$\sigma_{retro} < %s \cdot sigma_{retro mean}$'%k_threshold, size = 20)
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


result = plot_data.sort_values('energy_log10')
pred = result[dynedge].reset_index(drop = True)
true = result[variable].reset_index(drop = True)
retro_pred = result[retro].reset_index(drop = True)
E = result['energy_log10'].reset_index(drop = True)
num_bins = 10
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
axs_perf[0,1].set_title('$\sigma_{retro} < %s \cdot sigma_{retro mean}$'%k_threshold, size = 20)
axs_perf[1,1].errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
axs_perf[1,1].errorbar(means_E,list(width_retro),errors_width_retro,linestyle='dotted',fmt = 'o',capsize = 10, label = 'retro')
axs_perf[1,1].legend()
axs_perf[1,1].grid()
axs_perf[1,1].set_xlabel('$Energy_{log_{10}} GeV$', size = 20) 

plt.show()
