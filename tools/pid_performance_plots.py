import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def CalculateBias(results, variable, const):
    dynedge = variable + '_pred'
    result = results.sort_values('energy_log10')
    pred = result[dynedge].reset_index(drop = True)*const
    true = result[variable].reset_index(drop = True)*const
    #retro_pred = result[retro].reset_index(drop = True)
    E = result['energy_log10'].reset_index(drop = True)
    num_bins = 10

    n, bins_e, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
    plt.close()
    means_E = list()
    medians_error = list()

    errors_width = list()

    for k in range(len(bins_e)-1):
        index = (E >= bins_e[k]) & (E <= bins_e[k+1])
        if(sum(index) != 0):
            means_E.append(np.mean(E[index]))
            medians_error.append(np.median(pred[index]-true[index]))

            diff = (pred - true)[index].reset_index(drop = True)

            if variable == 'azimuth':
                diff[diff > np.pi*const] = diff[diff > np.pi*const] -np.pi*const
                diff[diff < -np.pi*const] = diff[diff < -np.pi*const] +np.pi*const

            x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
            x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)

            N = sum(index)
            fe_25 = sum(diff <= diff[x_25])/N
            fe_75 = sum(diff <= diff[x_75])/N
            errors_width.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25**2 + 1/fe_75**2))*(1/1.349))
            
            if( k == 0):
                errors = np.array([np.median(diff) - diff[x_25], 
                                   np.median(diff) - diff[x_75]])
                width = np.array(-diff[x_25]+ diff[x_75])/1.349
                

            else:
                errors = np.c_[errors,np.array([np.median(diff) - diff[x_25],
                                                np.median(diff) - diff[x_75]])]
                width = np.r_[width,np.array(-diff[x_25]+ diff[x_75])/1.349]
                 
    return errors, width, errors_width,means_E
def MakePlots(results, variable, scaler):
    if variable == 'zenith' or variable == 'azimuth':
        const = 360/(np.pi*2)
        fig = plt.figure()
        error_dict = {}
        width_dict = {}
        error_width_dict = {}
        E_dict = {}
        for key in results.keys():
            result  = results[key]
            
            if variable != 'azimuth':
                result[variable] = scaler['truth'][variable].inverse_transform(np.array(result[variable]).reshape(-1,1))
                result[variable + '_pred'] = scaler['truth'][variable].inverse_transform(np.array(result[variable + '_pred']).reshape(-1,1))
            else:
                index = result[variable +'_pred']<0
                result[variable +'_pred'][index] = result[variable +'_pred'][index] + 2*np.pi
                index = result[variable +'_pred']>2*np.pi
                result[variable +'_pred'][index] = result[variable +'_pred'][index] - 2*np.pi
            result['energy_log10'] = scaler['truth']['energy_log10'].inverse_transform(np.array(result['energy_log10']).reshape(-1,1))
            
        
            errors, width, errors_width, E = CalculateBias(result, variable, const)
            error_dict[key] = errors
            width_dict[key] = width
            error_width_dict[key] = errors_width
            E_dict[key] = E
        for key in results.keys():
            if key == '12':
                label = '$\\nu_e$: %s mill.'%(round(len(results[key])/1000000,2))
            if key == '14':
                label = '$\\nu_{\\mu}$: %s mill.'%(round(len(results[key])/1000000,2))
            if key == '16':
                label = '$\\nu_{\\tau}$: %s mill.'%(round(len(results[key])/1000000,2))
            if 'mix' in key:
                if '16' in key:
                    label = '($\\nu_e$, $\\nu_{\\mu}$, $\\nu_{\\tau}$)$|_{\\nu_{\\tau}}$: %s mill.'%(round(len(results[key])/1000000,2))
                
                if '14' in key:
                    label = '($\\nu_e$, $\\nu_{\\mu}$, $\\nu_{\\tau}$)$|_{\\nu_{\\mu}}$: %s mill.'%(round(len(results[key])/1000000,2))
                    
                if '12' in key:
                    label = '($\\nu_e$, $\\nu_{\\mu}$, $\\nu_{\\tau}$)$|_{\\nu_e}$: %s mill.'%(round(len(results[key])/1000000,2))
                    
            plt.errorbar(E_dict[key],width_dict[key],error_width_dict[key],linestyle='dotted',fmt = 'o',capsize = 10,label = label)
        plt.legend()
        plt.ylabel('W($\\Delta$ ø) [Deg.]', size = 25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        plt.xlabel('$Energy_{log_{10}}$ [GeV]', size = 25)
    if variable == 'energy_log10':
        const = 1
        fig = plt.figure()
        error_dict = {}
        width_dict = {}
        error_width_dict = {}
        E_dict = {}
        for key in results.keys():
            result  = results[key]
            result[variable] = scaler['truth'][variable].inverse_transform(np.array(result[variable]).reshape(-1,1))
            result[variable + '_pred'] = scaler['truth'][variable].inverse_transform(np.array(result[variable + '_pred']).reshape(-1,1))
            #result['energy_log10'] = scaler['truth']['energy_log10'].inverse_transform(np.array(result['energy_log10']).reshape(-1,1))
            
            errors, width, errors_width, E = CalculateBias(result, variable, const)
            error_dict[key] = errors
            width_dict[key] = width
            error_width_dict[key] = errors_width
            E_dict[key] = E
        for key in results.keys():
            if key == '12':
                label = '$\\nu_e$: %s mill.'%(round(len(results[key])/1000000,2))
            if key == '14':
                label = '$\\nu_{\\mu}$: %s mill.'%(round(len(results[key])/1000000,2))
            if key == '16':
                label = '$\\nu_{\\tau}$: %s mill.'%(round(len(results[key])/1000000,2))
            if 'mix' in key:
                if '16' in key:
                    label = '($\\nu_e$, $\\nu_{\\mu}$, $\\nu_{\\tau}$)$|_{\\nu_{\\tau}}$: %s mill.'%(round(len(results[key])/1000000,2))
                
                if '14' in key:
                    label = '($\\nu_e$, $\\nu_{\\mu}$, $\\nu_{\\tau}$)$|_{\\nu_{\\mu}}$: %s mill.'%(round(len(results[key])/1000000,2))
                    
                if '12' in key:
                    label = '($\\nu_e$, $\\nu_{\\mu}$, $\\nu_{\\tau}$)$|_{\\nu_e}$: %s mill.'%(round(len(results[key])/1000000,2))
                    
            plt.errorbar(E_dict[key],width_dict[key],error_width_dict[key],linestyle='dotted',fmt = 'o',capsize = 10,label = label)
        plt.legend()
        
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
       
        plt.xlabel('$Energy_{log_{10}}$ [GeV]', size = 25)
        plt.ylabel('$w(log_{10}(\\frac{E_{pred}}{E_{true}}))$', size = 20)
    plt.title('PID Comparison: %s'%variable,size = 20)
    return
# zenith
#all_e = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_e_v2\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
#all_t = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_tau\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
#all_mu = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\2mio_muons_only\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
#mix = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\mix_2mio\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
#everything = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e-check\results.csv')

# energy_log10
all_e = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_e_v2\dynedge-protov2-energy-k=8-c3not3-thesis-oldlr-50e-check-noprob\results.csv')
all_t = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_tau\dynedge-protov2-energy-k=8-c3not3-thesis-newlr-30e-noprob\results.csv')
all_mu = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\2mio_muons_only\dynedge-protov2-energy-k=8-c3not3-extrafeats-noprob\results.csv')
#mix = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\mix_2mio\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
everything = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-protov2-energy-k=8-c3not3-thesis-newlr-30e-check-noprob\results.csv')

# azimuth
#all_e = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_e_v2\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e-check\results.csv')
#all_t = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\all_tau\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e\results.csv')
#all_mu = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\2mio_muons_only\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e\results.csv')
#mix = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\mix_2mio\dynedge-protov2-zenith-k=8-c3not3-thesis-newlr-30e\results.csv')
#everything = pd.read_csv(r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything\dynedge-azimuth-protov2-c3not3-k=8-thesis-newlr-30e-check\results.csv')



mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'
scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')

with sqlite3.connect(mc_db) as con:
    query = 'select event_no, pid from truth where event_no in %s'%(str(tuple(everything['event_no'])))
    truth = pd.read_sql(query,con)

everything = everything.sort_values('event_no').reset_index(drop = True)
truth = truth.sort_values('event_no').reset_index(drop = True)

mix_16 = everything.loc[abs(truth['pid'] == 16),:]
mix_14 = everything.loc[abs(truth['pid'] == 14),:]
mix_12 = everything.loc[abs(truth['pid'] == 12),:]


results = {'12': all_e, '14': all_mu, '16': all_t,'mix_12': mix_12,'mix_14': mix_14, 'mix_16': mix_16}

variable = 'energy_log10'

MakePlots(results, variable, scalers)

