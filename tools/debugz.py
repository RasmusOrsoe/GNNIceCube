import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.backends.backend_pdf

db_file = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'

scalers_dyn = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')

path0 = r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything_zenithw\dynedge-protov2-zenith-k=8-c3not3-thesis-50epoch-weighted\results.csv'
path1 = r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything_zenithw\dynedge-protov2-zenith-k=8-c3not3-thesis-50epoch-weighted-x3-newnorm\results.csv'
path2 = r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything_zenithw\dynedge-protov2-zenith-k=8-c3not3-thesis-50epoch-weighted-x50-newlr\results.csv'
path3 = r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything_zenithw\dynedge-protov2-zenith-k=8-c3not3-thesis-50epoch-weighted-x3\results.csv'

res0 = pd.read_csv(path0).sort_values('event_no').reset_index(drop = True)
res1 = pd.read_csv(path1).sort_values('event_no').reset_index(drop = True)
res2 = pd.read_csv(path2).sort_values('event_no').reset_index(drop = True)
res3 = pd.read_csv(path3).sort_values('event_no').reset_index(drop = True)


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

if variable == 'azimuth':
    const = 360/(2*np.pi)
    bins = np.arange(0,6.1,0.1)*const
if variable == 'zenith':
    const = 360/(2*np.pi)
    bins = np.arange(0,3.1,0.1)*const
if variable == 'energy_log10':
    const = 1
    bins = np.arange(0,4,0.1)



plot_data = pd.concat( [pd.DataFrame(res0['event_no']),
                           pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(res0[variable]).reshape(-1,1))*const),
                           pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(res0[dynedge]).reshape(-1,1))*const),
                           pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(res1[dynedge]).reshape(-1,1))*const),
                           pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(res2[dynedge]).reshape(-1,1))*const),
                           pd.DataFrame(scalers_dyn['truth'][variable].inverse_transform(np.array(res3[dynedge]).reshape(-1,1))*const),
                           pd.DataFrame(1/np.sqrt(abs(res0[sigma]))*const),
                           pd.DataFrame(1/np.sqrt(abs(res1[sigma]))*const),
                           pd.DataFrame(1/np.sqrt(abs(res2[sigma]))*const),
                           pd.DataFrame(1/np.sqrt(abs(res3[sigma]))*const)],axis = 1)
plot_data.columns = ['event_no',variable,'reco0','reco1','reco2','reco3', 'sigma0','sigma1','sigma2','sigma3']
events = plot_data['event_no']
with sqlite3.connect(db_file) as con:
    query = 'select event_no, pid from truth where event_no in %s'%(str(tuple(events)))
    data = pd.read_sql(query,con)
data.sort_values('event_no').reset_index(drop = True)    
plot_data['pid'] = data['pid']


####
#
# Real K cut
#
#####

n = int(len(plot_data)*k_threshold)
pids = [12,14,16]
figs = []
fig = plt.figure()
reco = 'reco0'
sigma = 'sigma0'
plot_data_cut = plot_data.nsmallest(n, sigma).sort_values('event_no')
recos = ['reco0','reco1','reco2','reco3']
for i in range(0,4):
    fig = plt.figure()
    plt.title('%s'%(i))
    sigma = 'sigma' + str(i)
    k = 'reco' + str(i)
    plot_data_cut = plot_data.nsmallest(n, sigma).sort_values('event_no')
    plt.hist(plot_data_cut[k], bins = bins, histtype = 'step', label = '%s reco'%k, density = True)
    plt.hist(plot_data_cut[variable], bins =  bins, histtype = 'step', label = '%s truth'%k, density = True)
#%%
for pid in pids:
    pid_index = abs(plot_data['pid']) == pid
    plt.title('%s'%(reco))
    plt.hist(plot_data_cut[reco][pid_index], bins = bins, histtype = 'step', label = '%s reco'%pid, density = True)
    plt.hist(plot_data_cut[variable][pid_index], bins =  bins, histtype = 'step', label = '%s truth'%pid, density = True)
    plt.legend()
    figs.append(fig)
    
#%%
fig2 = plt.figure()
plt.hist2d(plot_data_cut[sigma],plot_data_cut[variable] - plot_data_cut[reco], bins= 100)
plt.xlabel('sigma')
plt.ylabel('angle diff')
figs.append(fig2)
fig3 = plt.figure()
plt.hist2d(plot_data_cut[variable],plot_data_cut[variable] - plot_data_cut[reco], bins= 100)
plt.xlabel('zenith')
plt.ylabel('angle diff')
figs.append(fig3)
fig4 = plt.figure()
plt.hist2d(plot_data_cut[variable],plot_data_cut[sigma], bins= 100)
plt.xlabel('zenith')
plt.ylabel('sigma')
figs.append(fig4)
fig5 = plt.figure()
plt.hist2d(plot_data_cut[variable],(plot_data_cut[variable] - plot_data_cut[reco])/(2*plot_data_cut[sigma]), bins= 100)
plt.plot(np.arange(0,180,0.01), np.repeat(1, len(np.arange(0,180,0.01))), lw = 3, color = 'red')
plt.plot(np.arange(0,180,0.01), np.repeat(-1, len(np.arange(0,180,0.01))), lw = 3, color = 'red')
plt.xlabel('zenith')
plt.ylabel('pull')
figs.append(fig5)

bands = np.arange(0,190,10)
fig6 = plt.figure()
for i in range(1,len(bands)):
    index = (plot_data_cut[variable] >= bands[i-1]) & (plot_data_cut[variable] <= bands[i])
    zenith = plot_data_cut[variable][index]
    pull = (plot_data_cut[variable] - plot_data_cut[reco])/(2*plot_data_cut[sigma])
    x = (plot_data_cut[sigma]/pull)[index]
    plt.scatter(x,zenith)
plt.xlim(-5,100)
plt.ylabel(variable)
plt.xlabel('sigma/pull')
figs.append(fig6)

fig7 = plt.figure()
index = (plot_data_cut[variable] >= 120) & (plot_data_cut[variable] <= 160)
zenith = plot_data_cut[variable][index]
pull = (plot_data_cut[variable] - plot_data_cut[reco])/(2*plot_data_cut[sigma])
x = (plot_data_cut[sigma]/pull)[index]
plt.hist2d(x,zenith, bins = [np.arange(-100,100,1),np.arange(120,160,1)])
#plt.xlim(-5,100)
plt.ylabel(variable)
plt.xlabel('sigma/pull')
figs.append(fig7)
pdf_nam = 'hilfe'

k = 0
#for figz in figs:
#    pdf_nam = 'hilfe' +str(k)
#    pdf = matplotlib.backends.backend_pdf.PdfPages("X:\\speciale\\%s.pdf"%pdf_nam)
#    pdf.savefig( figz, bbox_inches='tight' )
#    pdf.close()
#    k +=1