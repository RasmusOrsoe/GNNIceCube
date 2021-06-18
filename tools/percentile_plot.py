import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
def GrabMC(events):
    mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_000\data\dev_level7_mu_e_tau_oscweight_000.db'
    with sqlite3.connect(mc_db) as con:
        query = 'select * from truth where event_no in %s'%(str(tuple(events)))
        truth = pd.read_sql(query,con)
    
    retro_mc = truth.sort_values('event_no').reset_index(drop = True)
    return retro_mc

def CalculatePull(results, retro_mc, variable, const):
    dynedge = variable + '_pred'
    retro = variable + '_retro'
    result = results.sort_values('energy_log10')
    pred = result[dynedge].reset_index(drop = True)
    true = result[variable].reset_index(drop = True)
    true_retro = retro_mc[variable].reset_index(drop = True)
    retro_pred = retro_mc[retro].reset_index(drop = True)
    k = 2
    eps =0
    pull_retro = (retro_pred - true_retro)/(retro_mc[variable + '_sigma'])
    
    
    pull_dynedge = (pred - true)/(results[dynedge + '_k']*k)
    
    unit = np.random.normal(0, 1, len(pull_dynedge))
    
    fig = plt.figure()
    plt.hist2d(pred,true, bins = 100)
    plt.title(variable)
    
    running_pull_dynedge = []
    running_pull_retro = []
    plot_quantiles = []
    
    fig = plt.figure()
    pcp =  np.arange(0,1,0.01)
    percentiles = np.quantile(results[dynedge + '_k']*k,pcp)
    
    for i in range(1,len(percentiles)):
        plot_quantiles.append(np.mean(pcp[i-1] + pcp[i]))
        
        index_dynedge = (percentiles[i-1] <= (pred - true)) & ( (pred - true) <= percentiles[i])
        running_pull_dynedge.append(np.std((pred[index_dynedge] - (true[index_dynedge]))/(results[dynedge + '_k'][index_dynedge]*k)))




    percentiles = np.quantile(retro_mc[variable + '_sigma']*k, pcp)
    
        
    for i in range(1,len(percentiles)):

        index_retro   = (percentiles[i-1] <= (retro_pred - true_retro)) & ( (retro_pred - true_retro)  <= percentiles[i])
        retro_roll = ((retro_pred[index_retro] - true_retro[index_retro])/(retro_mc[variable + '_sigma'][index_retro]*k))
        retro_roll = retro_roll[~np.isinf(retro_roll)]
        running_pull_retro.append(np.std(retro_roll[abs(retro_roll) < 5]))


        
    
             
    return pull_retro,pull_dynedge, unit, running_pull_dynedge, running_pull_retro,  plot_quantiles
def MakePlots(results, scaler):

    const = 360/(np.pi*2)
    
    bins = np.linspace(-10,10,600)

    for key in results.keys():
        fig = plt.figure()
        variable = key
        
        result  = results[key]
        print(result.columns)
        retro_mc = GrabMC(result['event_no'])
        
        if variable != 'azimuth':
            result[variable] = scaler['truth'][variable].inverse_transform(np.array(result[variable]).reshape(-1,1))
            result[variable + '_pred'] = scaler['truth'][variable].inverse_transform(np.array(result[variable + '_pred']).reshape(-1,1))
            #result[variable + '_pred_k'] = 1/np.sqrt((abs(result[variable + '_pred_k'])))
            result[variable + '_pred_k'] = 1/np.sqrt((scaler['truth'][variable].inverse_transform(np.array(abs(result[variable + '_pred_k'])).reshape(-1,1))))
        if variable == 'azimuth':
            #fig = plt.figure()
            #plt.hist2d(result['azimuth'],result['azimuth_pred'], bins = 100)
            #plt.title(variable)
            index = result[variable +'_pred']<0
            result[variable +'_pred'][index] = result[variable +'_pred'][index] + 2*np.pi
            index = result[variable +'_pred']>2*np.pi
            result[variable +'_pred'][index] = result[variable +'_pred'][index] - 2*np.pi
            result[variable + '_pred_k'] = 1/np.sqrt(abs(result[variable + '_pred_k']))
        retro_mc[variable] = scaler['truth'][variable].inverse_transform(np.array(retro_mc[variable]).reshape(-1,1))
        #result['energy_log10'] = scaler['truth']['energy_log10'].inverse_transform(np.array(result['energy_log10']).reshape(-1,1))
        
    
        pull_retro, pull_dynedge, unit,  running_pull_dynedge, running_pull_retro, percentiles = CalculatePull(result, retro_mc, variable, const)
        
        pull_retro = pull_retro[~np.isnan(pull_retro)]
        pull_dynedge = pull_dynedge[~np.isnan(pull_dynedge)]
        pull_retro = pull_retro[~np.isinf(pull_retro)]
        pull_dynedge = pull_dynedge[~np.isinf(pull_dynedge)]
        
        plt.hist(pull_retro, histtype = 'step', label = 'retro', bins = bins)
        plt.hist(pull_dynedge, histtype = 'step', label = 'dynedge', bins = bins)
        plt.hist(unit, histtype = 'step', alpha = 0.5, label = 'unit gauss',bins = bins)
        plt.legend()
        plt.title(variable)
        #plt.ylabel('W($\\Delta$ ø) [Deg.]', size = 25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        
        fig = plt.figure()
        plt.scatter(percentiles,running_pull_retro, label = 'retro')
        plt.scatter(percentiles, running_pull_dynedge, label ='dynedge')
        plt.legend()
        plt.ylim([-2,5])
        #plt.xlabel('$Energy_{log_{10}}$ [GeV]', size = 25)

    
    return
# zenith
#all_e = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_e_v2\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
#all_t = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_tau\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
#all_mu_zenith = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\2mio_muons_only\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
all_mu_zenith = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\1mio_mu_only_zenith_even\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
mix = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_000\event_only_level7_all_oscweight_muon\dynedge-protov2-zenith-k=8-c3not3\results.csv')
everything_zenith = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e-check\results.csv')

#mix_zenith = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_000\event_only_level7_all_oscweight_1mio\dynedge-protov2-zenith-k=8-c3not3\results.csv')

# azimuth
#all_e = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_e_v2\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e-check\results.csv')
#all_t = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_tau\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e\results.csv')
all_mu_azimuth = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\2mio_muons_only\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e\results.csv')
#mix = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\mix_2mio\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
everything_azimuth = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e-check\results.csv')

#mix_azimuth = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_000\event_only_level7_all_oscweight_1mio\dynedge-azimuth-protov2-k-c3not3=8\results.csv')


scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_000\meta\transformers.pkl')



results = {'zenith': mix} #, 'azimuth': everything_azimuth}

MakePlots(results, scalers)

