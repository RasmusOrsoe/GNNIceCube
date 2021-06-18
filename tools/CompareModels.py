# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from time import strftime,gmtime
from copy import deepcopy

def GetResults(path):
    model_names = os.listdir(path)
    results = list()
    model_name_res = list()
    for name in model_names:
        if name != 'skip':
            model_res = pd.read_csv(path + "\\%s\\results.csv"%name)
            if 'scale' in name:
                if 'azimuth' in name:
                    print(name)
                    scaler = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')
                    index = model_res['azimuth']<0
                    model_res['azimuth'][index] = model_res['azimuth'][index] + 2*np.pi
                    index = model_res['azimuth']>2*np.pi
                    model_res['azimuth'][index] =model_res['azimuth'][index] - 2*np.pi
                    model_res['azimuth_pred'] = scaler['truth']['azimuth'].transform(np.array(model_res['azimuth_pred']).reshape(-1,1))
                if 'zenith' in name:
                    print(name)
                    scaler = pd.read_pickle(r'X:\speciale\data\rawdev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')
                    #index = model_res['zenith']<0
                    #model_res['zenith'][index] = model_res['zenith'][index] + 2*np.pi
                    #index = model_res['zenith']>2*np.pi
                    #model_res['zenith'][index] =model_res['zenith'][index] - 2*np.pi
                    model_res['zenith_pred'] = scaler['truth']['zenith'].transform(np.array(model_res['zenith_pred']).reshape(-1,1))
            results.append(model_res)
            model_name_res.append(name)
    
    return model_name_res,results

def SortIt(results,models,key):
    results_out = []
    models_out  = []
    for count in range(len(results)):
        if (sum(results[count].columns == key) != 0):
            models_out.append(models[count])
            results_out.append(results[count])
    return models_out,results_out
    
def MakePerformancePlots(results,scaler,scaler_E,models,mode):
    
    figs = list()
    error_list = list()
    means_list = list()
    rad_to_degrees = 360/(2*np.pi) 
    
    print('Making %s Performance Plots..'%mode)
    for i in range(0,len(results)):
        result = results[i]
        result = result.sort_values('event_no').reset_index(drop=True)
        result = result.sort_values('energy_log10')
        if mode == 'E':
            #E_pred = scaler.inverse_transform(np.array(result.loc[:,'energy_log10_pred']).reshape(-1,1))
            E_pred = np.array(result['energy_log10_pred'])
            E = np.array(result['energy_log10'])
            #E = scaler.inverse_transform(np.array(result.loc[:,'energy_log10']).reshape(-1,1))
            title  = '%s vs %s' %('E','$E_{pred}$')
            xlabel = 'log(E) [GeV]'
        if mode == 'azimuth':
            #if 'scale' not in models[i]:
            E_pred = scaler.inverse_transform(np.array(result.loc[:,'azimuth_pred']).reshape(-1,1))*rad_to_degrees
            E = scaler.inverse_transform(np.array(result.loc[:,'azimuth']).reshape(-1,1))*rad_to_degrees
            title  = '%s vs %s' %('Azimuth','$Azimuth_{pred}$')
            xlabel = 'Azimuth [Deg.]'
        if mode == 'zenith':
            E_pred = scaler.inverse_transform(np.array(result.loc[:,'zenith_pred']).reshape(-1,1))*rad_to_degrees
            E = scaler.inverse_transform(np.array(result.loc[:,'zenith']).reshape(-1,1))*rad_to_degrees
            title  = '%s vs %s' %('Zenith','$Zenith_{pred}$')
            xlabel = 'Zenith [Deg.]'
        num_bins = 50
        
        n, bins, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
        plt.close()
        error = list()
        means = list()
        means_E = list()
        for k in range(len(bins)-1):
            index_pred = (E_pred >= bins[k]) & (E_pred <= bins[k+1])
            index = (E >= bins[k]) & (E <= bins[k+1])
            means.append(np.mean(E_pred[index]))
            means_E.append(np.mean(E[index]))
            error.append(np.mean(abs(E[index] - E_pred[index])))
            #if mode == 'azimuth':
            #    if sum(abs(E[index] - E_pred[index]) > 360) > 0:
            #        print('wwwwuuuu')
            #        angdiff = abs(E[index] - E_pred[index])
            #        angdiff_index = angdiff >360
            #        angdiff[angdiff_index] = angdiff[angdiff_index] -360 
            #        error.append(np.mean(abs(E[index] - E_pred[index])))
        error_list.append(error)
        means_list.append(means)
        
        
        
    fig2=plt.figure()
    host2 = fig2.add_subplot(111)
    par2 = host2.twinx()
    host2.set_xlabel(xlabel,size = 20)
    host2.set_ylabel('%s$_{pred}$'%mode, size = 20)
    par2.set_ylabel('Count',size = 20)
    for j in range(0,len(means_list)):             
        host2.errorbar(means_E,means_list[j],error_list[j],linestyle='dotted',fmt = 'o',capsize = 15)
    num_bins = 50
    n, bins, patches = par2.hist(E, num_bins, facecolor='blue', alpha=0.1,label = '_nolegend_')   
    host2.plot(means_E,means_E,label = '_nolegend_')
    host2.legend(models)
    plt.title(title, size = 20)
    #host2.axis((1,4,1,4))
    figs.append(fig2)
    plt.close()
    
    
    
    means_log_list = []
    medians_E_list = []
    error_list  = []
    for i in range(0,len(results)):
        result = results[i]
        result = result.sort_values('event_no').reset_index(drop=True)
        result = result.sort_values('energy_log10')
        if mode == 'E':
            #E_pred_descaled = scaler.inverse_transform(np.array(result.loc[:,'energy_log10_pred']).reshape(-1,1))
            #E_descaled = scaler_E.inverse_transform(np.array(result.loc[:,'energy_log10']).reshape(-1,1))
            E_pred_descaled = np.array(result['energy_log10_pred'])
            E_descaled = np.array(result['energy_log10'])
        if mode == 'azimuth':
           # if 'scale' not in models[i]:
            E_pred_descaled = scaler.inverse_transform(np.array(result.loc[:,'azimuth_pred']).reshape(-1,1))*rad_to_degrees
            true  = scaler.inverse_transform(np.array(result.loc[:,'azimuth']).reshape(-1,1))*rad_to_degrees
            E_descaled = scaler_E.inverse_transform(np.array(result.loc[:,'energy_log10']).reshape(-1,1))
        if mode == 'zenith':
            E_pred_descaled = scaler.inverse_transform(np.array(result.loc[:,'zenith_pred']).reshape(-1,1))*rad_to_degrees
            true  = scaler.inverse_transform(np.array(result.loc[:,'zenith']).reshape(-1,1))*rad_to_degrees
            E_descaled = scaler_E.inverse_transform(np.array(result.loc[:,'energy_log10']).reshape(-1,1))
        E_pred = E_pred_descaled #result.loc[:,'E_pred']
        E = E_descaled#result.loc[:,'E']
        num_bins = 10
        fig3 = plt.figure()
        n, bins, patches = plt.hist(E_descaled, num_bins, facecolor='blue', alpha=0.3,label = None)
        plt.close()
        means_log = list()
        means_E = list()
        medians_E = list()
        for k in range(len(bins)-1):
            index = (E_descaled >= bins[k]) & (E_descaled <= bins[k+1])
            if(sum(index) != 0):
                means_log.append(np.mean(E_pred_descaled[index]-E_descaled[index]))
                means_E.append(np.mean(E_descaled[index]))
                if mode == 'E':
                    medians_E.append(np.median(E_pred_descaled[index]-E_descaled[index]))
                    diff = (E_pred_descaled - E_descaled)[index]
                    ylabel = r'$log(\frac{E_{pred}}{E}$)'
                    title = r' $log(\frac{E_{pred}}{E}$) vs E'
                if mode == 'azimuth' or mode == 'zenith':
                    medians_E.append(np.median(E_pred_descaled[index]-true[index]))
                    diff = (E_pred_descaled -true)[index]
                    ylabel = r'$%s_{pred} - %s$ [Deg.]'%(mode,mode)
                    title = r' $%s_{pred}$ vs E'%mode
                N = sum(index)
                
                x_16 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
                x_84 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
                
                if( k == 0):
                    errors = np.array([np.median(diff) - diff[x_16], 
                                       np.median(diff) - diff[x_84]])
                else:
                    errors = np.c_[errors,np.array([np.median(diff) - diff[x_16],
                                                    np.median(diff) - diff[x_84]])]
            
        means_log_list.append(means_log)
        medians_E_list.append(medians_E)
        error_list.append(errors)    
            
    keep_me = deepcopy(error_list)    
    fig=plt.figure()
    host = fig.add_subplot(111)
    par1 = host.twinx()
    host.set_xlabel('log(E) [GeV]',size = 20)
    host.set_ylabel(ylabel, size = 20)
    par1.set_ylabel('Count',size = 20)
    n, bins, patches = par1.hist(E_descaled, num_bins, facecolor='blue', alpha=0.1,label = '_nolegend_')
    for j in range(0,len(means_log_list)):            
        #host.scatter(means_E,means_log_list[j],color = 'red', s = 30,label = '_nolegend_')
        host.errorbar(means_E,medians_E_list[j],abs(error_list[j]),linestyle='dotted',fmt = 'o',capsize = 10)
    host.legend(models)
    #host.set_ylim([-1.30,0.6])
    host.grid()
    plt.title(title, size = 20)
    figs.append(fig)
    plt.close()
    
    error_list = []
    means_list = []
    width_list = []
    for i in range(0,len(results)):
        result = results[i]
        result = result.sort_values('event_no').reset_index(drop=True)
        result = result.sort_values('energy_log10')
        if mode == 'E':
            #E_pred_descaled = scaler.inverse_transform(np.array(result.loc[:,'energy_log10_pred']).reshape(-1,1))
            #E_descaled =  scaler.inverse_transform(np.array(result.loc[:,'energy_log10']).reshape(-1,1))
            E_pred_descaled = np.array(result.loc[:,'energy_log10_pred'])
            E_descaled =  np.array(result.loc[:,'energy_log10'])
            ylabel = r'W($log(\frac{E_{pred}}{E}$))'
            title = r'W($log(\frac{E_{pred}}{E}$)) vs E'
        if mode == 'azimuth':
            #if 'scale' not in models[i]:
            E_pred_descaled = scaler.inverse_transform(np.array(result.loc[:,'azimuth_pred']).reshape(-1,1))*rad_to_degrees
            E_descaled = scaler.inverse_transform(np.array(result.loc[:,'azimuth']).reshape(-1,1))*rad_to_degrees
            ylabel = r'W($%s_{pred}$)'%mode
            title = r'W($%s_{pred}$) vs E'%mode
        if mode == 'zenith':
            E_pred_descaled = scaler.inverse_transform(np.array(result.loc[:,'zenith_pred']).reshape(-1,1))*rad_to_degrees
            E_descaled =  scaler.inverse_transform(np.array(result.loc[:,'zenith']).reshape(-1,1))*rad_to_degrees
            ylabel = r'W($%s_{pred}$ [Deg.])'%mode
            title = r'W($%s_{pred}$) vs E'%mode
        E_pred = E_pred_descaled
        E = E_descaled#scaler_E.inverse_transform(np.array(result.loc[:,'energy_log10']).reshape(-1,1))
        num_bins = np.arange(-0.5,4,0.5)
        fig3 = plt.figure()
        n, bins, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
        plt.close()
        means_log = list()
        means_E = list()
        medians_E = list()
        width = list()
        errors = list()
        for k in range(len(bins)-1):
            index = (E >= bins[k]) & (E <= bins[k+1])
            if(sum(index) != 0):
                means_log.append(np.mean(E_pred_descaled[index]-E_descaled[index]))
                means_E.append(np.mean(E[index]))
                medians_E.append(np.median(E_pred_descaled[index]-E_descaled[index]))
                N = sum(index)
                diff = (E_pred_descaled - E_descaled)[index]
                x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
                x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
                fe_25 = sum(diff <= diff[x_25])/N
                fe_75 = sum(diff <= diff[x_75])/N
                errors.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25**2 + 1/fe_75**2))*(1/1.349))
                if( k == 0):
                    width = np.array(-diff[x_25]+ diff[x_75])/1.349
                else:
                    width = np.r_[width,np.array(-diff[x_25]+ diff[x_75])/1.349]
        
        width_list.append(width)
        error_list.append(errors)
        means_list.append(means_E)
    fig7 = plt.figure()
    plt.close
    fig7=plt.figure()
    #host2 = plt.subplot2grid((4,3),(3,0),colspan =  3)
    #host2.yaxis.set_ticks(np.arange(0,1.10,0.20))
    #host2.grid()
    #plt.xlabel('Log(E) [GeV]',size = 15)
    #plt.ylabel('Rel. Imp.', size = 20)
    #host2.plot(means_E,np.repeat(0,len(means_E)),color = 'black')
    host = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=3)
    par1 = host.twinx()
    host.set_ylabel(ylabel, size = 15)
    host.set_xlabel('Log(E) [GeV])',size = 20)
    #host.tick_params(axis='both', which='major', labelsize=20)
    #host.tick_params(axis='both', which='minor', labelsize=20)
    par1.set_ylabel('Events',size = 20)
    n, bins, patches = par1.hist(E, bins = num_bins, facecolor='blue', alpha=0.1,label = None)
    # if mode == 'zenith':
    #      pd.DataFrame(means_list[0]).to_csv(r'X:\speciale\results\aachen_slides\aachen_slides\zenith_means.csv')
    #      pd.DataFrame(width_list[0]).to_csv(r'X:\speciale\results\aachen_slides\aachen_slides\zenith_width.csv')
    #      pd.DataFrame(error_list[0]).to_csv(r'X:\speciale\results\aachen_slides\aachen_slides\zenith_error.csv')
    # if mode == 'E':
    #      pd.DataFrame(means_list[0]).to_csv(r'X:\speciale\results\aachen_slides\aachen_slides\E_means.csv')
    #      pd.DataFrame(width_list[0]).to_csv(r'X:\speciale\results\aachen_slides\aachen_slides\E_width.csv')
    #      pd.DataFrame(error_list[0]).to_csv(r'X:\speciale\results\aachen_slides\aachen_slides\E_error.csv')
    for j in range(0,len(means_list)):            
         host.errorbar(means_list[j],list(width_list[j]),error_list[j],linestyle='dotted',fmt = 'o',capsize = 10)
    #if mode == 'zenith':
    #     error_hack = pd.read_csv(r'X:\speciale\results\aachen_slides\aachen_slides\skip\zenith_DC (SRT)\zenith_error.csv')['0']
    #     mean_hack  = pd.read_csv(r'X:\speciale\results\aachen_slides\aachen_slides\skip\zenith_DC (SRT)\zenith_means.csv')['0']
    #     width_hack = pd.read_csv(r'X:\speciale\results\aachen_slides\aachen_slides\skip\zenith_DC (SRT)\zenith_width.csv')['0']
    #     host.errorbar(mean_hack,list(width_hack),error_hack,linestyle='dotted',fmt = 'o',capsize = 10, color = 'grey')
    #     hack_tag = 'zenith_DC (SRT)'
    #if mode == 'E':
    #     error_hack = pd.read_csv(r'X:\speciale\results\aachen_slides\aachen_slides\skip\energy_DC (SRT)\E_error.csv')['0']
    #     mean_hack  = pd.read_csv(r'X:\speciale\results\aachen_slides\aachen_slides\skip\energy_DC (SRT)\E_means.csv')['0']
    #     width_hack = pd.read_csv(r'X:\speciale\results\aachen_slides\aachen_slides\skip\energy_DC (SRT)\E_width.csv')['0']
    #     host.errorbar(mean_hack,list(width_hack),error_hack,linestyle='dotted',fmt = 'o',capsize = 10, color = 'grey')
    #     hack_tag = 'energy_DC (SRT)'
    tags = []
    for model in models:
        tags.append(model)
   #tags.append(hack_tag)
    host.legend(tags)
    #host.yaxis.set_ticks(np.arange(0,1.1,0.1)) 
    host.set_xlim(-0.5,4)
    plt.title( title, size = 20)
    host.grid()
    figs.append(fig7)
    plt.close()
    for i in range(0,len(models)):
        result = results[i]
        if mode == 'E':
            E_pred = scaler.inverse_transform(np.array(result.loc[:,'energy_log10_pred']).reshape(-1,1))
            E = scaler.inverse_transform(np.array(result.loc[:,'energy_log10']).reshape(-1,1))
            #title  = '%s vs %s' %('E','$E_{pred}$')
            #xlabel = 'log(E) [GeV]'
            bins = np.arange(-0.5, 4,0.1)
            title = 'energy_log10 on "good cut"'
        if mode == 'azimuth':
            #if 'scale' not in models[i]:
            E_pred = scaler.inverse_transform(np.array(result.loc[:,'azimuth_pred']).reshape(-1,1))*rad_to_degrees
            E = scaler.inverse_transform(np.array(result.loc[:,'azimuth']).reshape(-1,1))*rad_to_degrees
            #title  = '%s vs %s' %('Azimuth','$Azimuth_{pred}$')
            #xlabel = 'Azimuth [Deg.]'
            bins = np.arange(0, 2*np.pi*rad_to_degrees,5)
            title = 'azimuth on "good cut"'
        if mode == 'zenith':
            E_pred = scaler.inverse_transform(np.array(result.loc[:,'zenith_pred']).reshape(-1,1))*rad_to_degrees
            E = scaler.inverse_transform(np.array(result.loc[:,'zenith']).reshape(-1,1))*rad_to_degrees
            #title  = '%s vs %s' %('Zenith','$Zenith_{pred}$')
            #xlabel = 'Zenith [Deg.]'
            bins = np.arange(0, np.pi*rad_to_degrees,5)
            title = 'zenith on "good cut"'
        fig_10 = plt.figure()
        plt.hist2d(E_pred.squeeze(), E.squeeze(), bins = bins)
        plt.title(title,size = 20)
        minval = bins.min()
        maxval = bins.max()
        plt.plot([minval,maxval],[minval,maxval], color = 'white')
        plt.xlim = [minval,maxval]
        plt.ylim = [minval,maxval]
        plt.ylabel('True', size = 20)
        plt.xlabel('Predicted', size = 20)
        figs.append(fig_10)
        plt.close()
    return figs    


path = r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\everything'

scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')

scaler_E = scalers['truth']['energy_log10']
scaler = scaler_E
models,results = GetResults(path)
models_E, results_E =  SortIt(results,models,'energy_log10_pred' )
a = 1

if len(models_E) != 0:
  figs_E = MakePerformancePlots(results_E, scaler,scaler_E,models_E,mode = 'E')

scaler = scalers['truth']['azimuth']
models,results = GetResults(path)
models_azimuth, results_azimuth =  SortIt(results,models,'azimuth_pred' ) 
if len(models_azimuth) != 0:
    figs_azimuth = MakePerformancePlots(results_azimuth, scaler,scaler_E, models_azimuth,mode = 'azimuth')

scaler = scalers['truth']['zenith']
models,results = GetResults(path)
models_zenith, results_zenith =  SortIt(results,models,'zenith_pred' ) 
if len(models_zenith) != 0:    
    figs_zenith = MakePerformancePlots(results_zenith, scaler,scaler_E, models_zenith,mode = 'zenith')

figs = []

if len(models_E) != 0:
    figs.extend(figs_E[:])
if len(models_azimuth) != 0:
    figs.extend(figs_azimuth[:])
if len(models_zenith) != 0: 
    figs.extend(figs_zenith[:])


pdf_nam = str([models])
if(len("X:\\speciale\\results\\runs\\results\\event_only\\reports\\%s.pdf"%pdf_nam) > 260):
    pdf_nam = str(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
pdf = matplotlib.backends.backend_pdf.PdfPages("X:\\speciale\\results\\runs\\results\\event_only\\reports\\%s.pdf"%pdf_nam)
for fig in figs: 
    pdf.savefig( fig, bbox_inches='tight' )
pdf.close()


