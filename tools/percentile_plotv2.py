import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
def GrabMC(events):
    mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'
    with sqlite3.connect(mc_db) as con:
        query = 'select * from truth where event_no in %s'%(str(tuple(events)))
        truth = pd.read_sql(query,con)
    
    retro_mc = truth.sort_values('event_no').reset_index(drop = True)
    return retro_mc


def gaussian(x, N, mu, sig):

    return N / (np.sqrt(2*np.pi) * sig) * np.exp(-0.5*((x - mu)/sig)**2) 


everything_zenith = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e-check\results.csv')

everything_azimuth = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e-check\results.csv')


scaler = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')



results = {'zenith': everything_zenith, 'azimuth': everything_azimuth} 





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

    retro_sigma = variable + '_sigma'
    retro = variable + '_retro'
    dynedge_sigma = variable + '_pred_k'
    dynedge = variable + '_pred'
    truth = variable
    
    
    diff_retro =  (retro_mc[retro]  -retro_mc[truth])
    diff_dynedge = (result[dynedge]-result[truth])
    
    if variable == 'azimuth':
        diff_dynedge[diff_dynedge > np.pi] = diff_dynedge[diff_dynedge > np.pi] -np.pi
        diff_dynedge[diff_dynedge < -np.pi] = diff_dynedge[diff_dynedge < -np.pi] +np.pi
        diff_retro[diff_retro > np.pi] = diff_retro[diff_retro > np.pi] -np.pi
        diff_retro[diff_retro < -np.pi] = diff_retro[diff_retro < -np.pi] +np.pi
    pull_retro   = diff_retro/retro_mc[retro_sigma]
    
    
    
    pull_dynedge = diff_dynedge/(result[dynedge_sigma]*2)
    
    
    
    std_pull_retro   = pull_retro[np.abs(pull_retro) < 8.0].std()
    
    std_pull_dynedge = pull_dynedge[np.abs(pull_dynedge) < 8.0].std()
    
    fig_MCpull, ax_MCpull = plt.subplots(figsize=(12,8))
    ax_MCpull.hist(pull_retro,   histtype='step', linewidth=2, color='red',  bins=100, range=(-10.0,10.0), label='Retro')
    ax_MCpull.hist(pull_dynedge, histtype='step', linewidth=2, color='blue', bins=100, range=(-10.0,10.0), label='DynEdge')
    ax_MCpull.hist(pull_retro / std_pull_retro,     histtype='step', linewidth=1, color='red',  bins=100, range=(-10.0,10.0), linestyle='dotted', label='Retro - scaled')
    ax_MCpull.hist(pull_dynedge / std_pull_dynedge, histtype='step', linewidth=1, color='blue', bins=100, range=(-10.0,10.0), linestyle='dotted', label='DynEdge - scaled')
    #ax_MCpull.set(xlabel="Pull distribution of Log10(Energy) estimates", ylabel="Frequency", title="")
    plt.xlabel("Pull distribution", fontsize = 20)
    plt.ylabel("Frequency", fontsize = 20)
    plt.title("Pull: %s"%variable, fontsize = 20)
    x_gauss = np.linspace(-10.0, 10.0, 200)
    ax_MCpull.plot(x_gauss, gaussian(x_gauss, N=len(retro_mc[retro])*0.2, mu=0.0, sig=1.0), color="black", linestyle="dotted", label='Unit Gaussian')
    
    ax_MCpull.text(-9.0, 38000.0, 'OscNext MC', fontsize=20, color='black')
    ax_MCpull.text(-9.0, 33000.0, f'N events = {len(retro_mc[retro]):6d}', fontsize=15, color='black')
    ax_MCpull.text(-9.0, 27000.0, f'Std(Retro pull) = {std_pull_retro:4.2f}', fontsize=15, color='black')
    ax_MCpull.text(-9.0, 20000.0, f'Std(DynEdge pull) = {std_pull_dynedge:4.2f}', fontsize=15, color='black')
    ax_MCpull.legend(fontsize = 18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    
    Nquantiles = 100

    means_retro  = []
    
    emeans_retro = []
    
    stds_retro   = []
    
    estds_retro  = []
    
    means_dynedge  = []
    
    emeans_dynedge = []
    
    stds_dynedge   = []
    
    estds_dynedge  = []
    
    
    
    Nsample = len(pull_retro) / Nquantiles
    
        
    
    for quant in range(Nquantiles) :
    
        Qlow_retro  = np.quantile(retro_mc[retro_sigma], quant    / Nquantiles)
    
        Qhigh_retro = np.quantile(retro_mc[retro_sigma],(quant+1) / Nquantiles)
    
        mask_retro = np.logical_and(Qlow_retro < retro_mc[retro_sigma], retro_mc[retro_sigma] < Qhigh_retro)
    
        means_retro.append(pull_retro[mask_retro].mean())
    
        emeans_retro.append(pull_retro[mask_retro].sem())
    
        stds_retro.append(pull_retro[mask_retro].std())
    
        estds_retro.append(pull_retro[mask_retro].std()/np.sqrt(2*Nsample))     # Barlow, page 79, eq. 5.23
    
        if (quant < 5) or (Nquantiles-quant < 5) :
    
            print(f"  Retro: {quant:2d}   mean = {means_retro[-1]:6.3f} +- {emeans_retro[-1]:5.3f}    std = {stds_retro[-1]:5.3f} +- {estds_retro[-1]:5.3f}")
    
    
    
        Qlow_dynedge  = np.quantile(result[dynedge_sigma]*2, quant    / Nquantiles)
    
        Qhigh_dynedge = np.quantile(result[dynedge_sigma]*2,(quant+1) / Nquantiles)
    
        mask_dynedge = np.logical_and(Qlow_dynedge < result[dynedge_sigma]*2, result[dynedge_sigma]*2 < Qhigh_dynedge)
    
        means_dynedge.append(pull_dynedge[mask_dynedge].mean())
    
        emeans_dynedge.append(pull_dynedge[mask_dynedge].sem())
    
        stds_dynedge.append(pull_dynedge[mask_dynedge].std())
    
        estds_dynedge.append(pull_dynedge[mask_dynedge].std()/np.sqrt(2*Nsample))     # Barlow, page 79, eq. 5.23
    
        if (quant < 5) or (Nquantiles-quant < 5) :
    
            print(f"  Dynedge: {quant:2d}   mean = {means_dynedge[-1]:6.3f} +- {emeans_dynedge[-1]:5.3f}    std = {stds_dynedge[-1]:5.3f} +- {estds_dynedge[-1]:5.3f}")
    
    
    
    
    
    x_quant = np.linspace(0.0, 1.0, Nquantiles)
    
    
    
    fig_MCquantpull, ax_MCquantpull = plt.subplots(figsize=(12,8))
    
    ax_MCquantpull.errorbar(x_quant, stds_retro,   yerr=estds_retro,   linewidth=2, color='red', label='Retro')
    
    ax_MCquantpull.errorbar(x_quant, stds_dynedge, yerr=estds_dynedge, linewidth=2, color='blue', label='DynEdge')
    
    ax_MCquantpull.set_ylim([0.0, 5.0])
    
    plt.xlabel("$\\sigma$ -percentiles", fontsize = 20)
    plt.ylabel("std of %s pulls"%variable, fontsize = 20)
    plt.title("std(Pull) vs $\\sigma$ -percentiles : %s"%variable, fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax_MCquantpull.legend(fontsize = 18)


