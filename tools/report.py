# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.backends.backend_pdf
import torch

path = r'J:\speciale\results\runs\results\event_only'
figs = list()
cutoff = 10



data_raw = 'dev_numu_train_upgrade_step4_2020_00' 

graphs   = 'dev_numu_train_upgrade_step4_2020_00\event_only_shuffled_input(0,1)_target-_2mio'
retro = False

scalers = pd.read_pickle(r'X:\speciale\data\raw\%s\meta\transformers.pkl'%data_raw)
scaler_mads = scalers['truth']['energy_log10']



for file in os.listdir(path): 
     if file.endswith(".csv"):
        cols = pd.read_csv(path + '\\' + file).sort_values('E').columns
nmae_cols = cols[0:1]         
pred_cols = cols[1:2]
true_cols = cols[2:3]
for c in range(0,len(true_cols)):
    for file in os.listdir(path): 
         if file.endswith(".csv"):
            result = pd.read_csv(path + '\\' + file).sort_values(true_cols[c])
            num_bins = 50
            n, bins, patches = plt.hist(result.loc[:,'E'], num_bins, facecolor='blue', alpha=0.3,label = None)
            plt.close()
            error = list()
            means_nmae = list()
            means_E = list()
            E_nmae = result.loc[:,'nmae E']
            E = result.loc[:,'E']
            for k in range(len(bins)-1):
                index = (E >= bins[k]) & (E <= bins[k+1])
                means_nmae.append(np.mean(E_nmae[index]))
                means_E.append(np.mean(E[index]))
                error.append(np.std(abs(E_nmae[index])))     
            fig=plt.figure()
            host = fig.add_subplot(111)
            par1 = host.twinx()
            host.set_xlabel(true_cols[c],size = 20)
            host.set_ylabel('NMAE E', size = 20)
            par1.set_ylabel('Count',size = 20)
            num_bins = 50
            n, bins, patches = par1.hist(result.loc[:,'E'], num_bins, facecolor='blue', alpha=0.01,label = None)            
            host.errorbar(means_E,means_nmae,error,linestyle='dotted',fmt = 'o',capsize = 15)
            host.plot(means_E,np.repeat(np.mean(result.loc[:,'nmae E']),50))
            host.legend(['avg. error %s '%round(np.mean(result.loc[:,'nmae E']),3),None])
            host.yaxis.set_ticks(np.arange(0,0.6,0.05))
            host.grid()
            plt.title('NMAE E vs E', size = 20)
            pdf_nam = file
            #figs.append(fig)
    
for c in range(0,len(true_cols)):
    for file in os.listdir(path): 
         if file.endswith(".csv"):
            result = pd.read_csv(path + '\\' + file)
            E_retro = pd.read_csv(r'J:\speciale\data\graphs\%s\retro_valid\retro.csv'%graphs)
            E_retro.drop_duplicates(subset ="event_no", keep = 'first', inplace = True) 
            E_retro = E_retro.sort_values('event_no').reset_index(drop=True)
            result = result.sort_values('event_no').reset_index(drop=True)
            E_retro[E_retro == -np.inf] = np.nan
            result[['retro','retro_events']] = E_retro
            result = result.sort_values('E')
            E_retro = np.array(result['retro']).reshape(-1,1)
            num_bins = 50
            E = np.array(result.loc[:,'E']).reshape(-1,1)
            n, bins, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
            plt.close()
            error = list()
            means = list()
            means_E = list()
            E_pred = np.array(result.loc[:,'E_pred']).reshape(-1,1)
            
            for k in range(len(bins)-1):
                index_pred = (E_pred >= bins[k]) & (E_pred <= bins[k+1])
                index = (E >= bins[k]) & (E <= bins[k+1])
                means.append(np.mean(E_pred[index]))
                means_E.append(np.mean(E[index]))
                error.append(np.mean(abs(E[index] - E_pred[index])))
            if(retro):
                error_retro = list()
                means_retro = list()
                means_E_retro = list()
                for k in range(len(bins)-1):
                    index_pred = (E_retro >= bins[k]) & (E_retro <= bins[k+1])
                    index = (E >= bins[k]) & (E <= bins[k+1])
                    means_retro.append(np.nanmean(E_retro[index]))
                    means_E_retro.append(np.mean(E[index]))
                    error_retro.append(np.nanmean(abs(E[index] - E_retro[index])))
                    print('RETRO: %s'%sum(index))
            fig2=plt.figure()
            host2 = fig2.add_subplot(111)
            par2 = host2.twinx()
            host2.set_xlabel(true_cols[c],size = 20)
            host2.set_ylabel(pred_cols[c], size = 20)
            par2.set_ylabel('Count',size = 20)
            num_bins = 50
            n, bins, patches = par2.hist(E, num_bins, facecolor='blue', alpha=0.1,label = None)            
            host2.errorbar(means_E,means,error,linestyle='dotted',fmt = 'o',capsize = 15)
            host2.errorbar(means_E_retro,means_retro,error_retro,linestyle='dotted',fmt = 'o',capsize = 15)
            host2.plot(means_E,means_E)
            host2.legend('dynedge GNN','retro')
            plt.title('%s vs %s' %(true_cols[c],pred_cols[c]), size = 20)
            #host2.axis((1,4,1,4))
            pdf_nam = file
            figs.append(fig2)
            plt.close()

for c in range(0,len(true_cols)):
    for file in os.listdir(path): 
         if file.endswith(".csv"):
            result = pd.read_csv(path + '\\' + file)
            E_retro = pd.read_csv(r'J:\speciale\data\graphs\%s\retro_valid\retro.csv'%graphs)
            E_retro.drop_duplicates(subset ="event_no", keep = 'first', inplace = True) 
            E_retro = E_retro.sort_values('event_no').reset_index(drop=True)
            result = result.sort_values('event_no').reset_index(drop=True)
            E_retro[E_retro == -np.inf] = np.nan
            result[['retro','retro_events']] = E_retro
            result = result.sort_values('E')
            E_retro = np.array(result['retro']).reshape(-1,1)
            E_pred_descaled = np.array(result.loc[:,'E_pred']).reshape(-1,1)
            E_descaled = np.array(result.loc[:,'E']).reshape(-1,1)
            E_pred = result.loc[:,'E_pred']
            E = result.loc[:,'E']
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
                    medians_E.append(np.median(E_pred_descaled[index]-E_descaled[index]))
                    N = sum(index)
                    diff = (E_pred_descaled - E_descaled)[index]
                    x_16 = abs(diff-np.percentile(diff,16,interpolation='nearest')).argmin() #int(0.16*N)
                    x_84 = abs(diff-np.percentile(diff,84,interpolation='nearest')).argmin() #int(0.84*N)
                    
                    if( k == 0):
                        errors = np.array([np.median(E_pred_descaled[index]-E_descaled[index]) - diff[x_16], 
                                           np.median(E_pred_descaled[index]-E_descaled[index]) - diff[x_84]])
                    else:
                        errors = np.c_[errors,np.array([np.median(E_pred_descaled[index]-E_descaled[index]) - diff[x_16],
                                                        np.median(E_pred_descaled[index]-E_descaled[index]) - diff[x_84]])]
                
                fig=plt.figure()
                axes= plt.axes()
                axes.set_xlim([-2.5,2.5])
                axes.tick_params(axis='x', labelsize= 5)
                axes.set_xticks(np.arange(-2.5,2.5,0.2))
                n_mini, bins_mini, patches_mini = plt.hist(E_pred_descaled[index]-E_descaled[index], 20, facecolor='blue', alpha=0.3,label = None)
                plt.title('dynedge GNN bin %s '%(k+1))
                plt.xlabel('log(E_pred/log_E)')
                plt.ylabel('Count')
                #ax = fig.axes()
                #ax.xaxis.set_ticks(np.arange(-2,2,0.1))
                plt.plot(np.repeat(diff[x_16],100),range(0,100),label = '16-pctil || %s'%str(round(diff[x_16],3)))
                plt.plot(np.repeat(diff[x_84],100),range(0,100),label = '84-pctil || %s'%str(round(diff[x_84],3)))
                plt.plot(np.repeat(np.median(E_pred_descaled[index]-E_descaled[index]),100),range(0,100),
                         label = '50-pctil || %s'%str(round(np.median(E_pred_descaled[index]-E_descaled[index]),3)),color='blue')
                plt.plot(np.repeat(np.mean(E_pred_descaled[index]-E_descaled[index]),100),range(0,100),
                         label = 'bin mean || %s'%str(round(np.mean(E_pred_descaled[index]-E_descaled[index]),3)),color='red')
                plt.legend()
                figs.append(fig)
                plt.close()
            if retro:
                means_log_retro = list()
                means_E_retro = list()
                medians_E_retro = list()
                for k in range(len(bins)-1):
                    index = (E_descaled >= bins[k]) & (E_descaled <= bins[k+1])
                    if(sum(index) != 0):
                        means_log_retro.append(np.nanmean(E_retro[index]-E_descaled[index]))
                        means_E_retro.append(np.nanmean(E_descaled[index]))
                        N = sum(index)
                        diff = (E_retro - E_descaled)[index]
                        diff = diff[np.logical_not(np.isnan(diff))]
                        medians_E_retro.append(np.median(diff))
                        x_16 = abs(diff-np.percentile(diff,16,interpolation='nearest')).argmin() #int(0.16*N)
                        x_84 = abs(diff-np.percentile(diff,84,interpolation='nearest')).argmin() #int(0.84*N)
                        
                        if( k == 0):
                            errors_retro = np.array([np.median(diff) - diff[x_16], 
                                               np.median(diff) - diff[x_84]])
                        else:
                            errors_retro = np.c_[errors_retro,np.array([np.median(diff) - diff[x_16],
                                                            np.median(diff) - diff[x_84]])]
                    
                    fig=plt.figure()
                    axes= plt.axes()
                    axes.set_xlim([-2.5,2.5])
                    axes.tick_params(axis='x', labelsize= 5)
                    axes.set_xticks(np.arange(-2.5,2.5,0.2))
                    n_mini, bins_mini, patches_mini = plt.hist(diff, 20, facecolor='blue', alpha=0.3,label = None)
                    plt.title('retro bin %s '%(k+1))
                    plt.xlabel('log(E_retro/log_E)')
                    plt.ylabel('Count')
                    #ax = fig.axes()
                    #ax.xaxis.set_ticks(np.arange(-2,2,0.1))
                    plt.plot(np.repeat(diff[x_16],100),range(0,100),label = '16-pctil || %s'%str(round(diff[x_16],3)))
                    plt.plot(np.repeat(diff[x_84],100),range(0,100),label = '84-pctil || %s'%str(round(diff[x_84],3)))
                    plt.plot(np.repeat(np.median(diff),100),range(0,100),
                            label = '50-pctil || %s'%str(round(np.median(diff),3)),color='blue')
                    plt.plot(np.repeat(np.mean(diff),100),range(0,100),
                             label = 'bin mean || %s'%str(round(np.mean(diff),3)),color='red')
                    plt.legend()
                    figs.append(fig)
                    plt.close()
                
            fig=plt.figure()
            host = fig.add_subplot(111)
            par1 = host.twinx()
            host.set_xlabel(true_cols[c],size = 20)
            host.set_ylabel('log(E_pred/E)', size = 20)
            par1.set_ylabel('Count',size = 20)
            n, bins, patches = par1.hist(E_descaled, num_bins, facecolor='blue', alpha=0.1,label = None)            
            host.scatter(means_E,means_log,color = 'red', s = 30)
            host.scatter(means_E_retro,means_log_retro,color = 'red', s = 30)
            host.errorbar(means_E,medians_E,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10)
            host.errorbar(means_E_retro,medians_E_retro,abs(errors_retro),linestyle='dotted',fmt = 'o',capsize = 10)
            #host.plot(means_E,np.repeat(np.mean(means_log),num_bins))
            host.legend([None,None,'dynedge bin median', 'retro bin median'])
            #host.yaxis.set_ticks(np.arange(-1,1.1,0.1))
            host.grid()
            plt.title(' log(E_pred/E) vs E', size = 20)
            pdf_nam = file
            figs.append(fig)
            
for c in range(0,len(true_cols)):
    for file in os.listdir(path): 
         if file.endswith(".csv"):
            result = pd.read_csv(path + '\\' + file)
            E_retro = pd.read_csv(r'J:\speciale\data\graphs\%s\retro_valid\retro.csv'%graphs)
            E_retro.drop_duplicates(subset ="event_no", keep = 'first', inplace = True) 
            E_retro = E_retro.sort_values('event_no').reset_index(drop=True)
            result = result.sort_values('event_no').reset_index(drop=True)
            E_retro[E_retro == -np.inf] = np.nan
            result[['retro','retro_events']] = E_retro
            result = result.sort_values('E')
            E_retro = np.array(result['retro']).reshape(-1,1)
            E_pred_descaled = np.array(result.loc[:,'E_pred']).reshape(-1,1)
            E_descaled = np.array(result.loc[:,'E']).reshape(-1,1)
            E_pred = result.loc[:,'E_pred']
            E = result.loc[:,'E']
            num_bins = 10
            fig3 = plt.figure()
            n, bins, patches = plt.hist(E_descaled, num_bins, facecolor='blue', alpha=0.3,label = None)
            plt.close()
            means_log = list()
            means_E = list()
            medians_E = list()
            width = list()
            errors = list()
            for k in range(len(bins)-1):
                index = (E_descaled >= bins[k]) & (E_descaled <= bins[k+1])
                if(sum(index) != 0):
                    means_log.append(np.mean(E_pred_descaled[index]-E_descaled[index]))
                    means_E.append(np.mean(E_descaled[index]))
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
            if retro:
                means_log_retro = list()
                means_E_retro = list()
                medians_E_retro = list()
                width_retro = list()
                errors_retro = list()
                for k in range(len(bins)-1):
                    index = (E_descaled >= bins[k]) & (E_descaled <= bins[k+1])
                    if(sum(index) != 0):
                        means_log_retro.append(np.nanmean(E_retro[index]-E_descaled[index]))
                        means_E_retro.append(np.mean(E_descaled[index]))
                        
                        N = sum(index)
                        diff = (E_retro - E_descaled)[index]
                        diff = diff[np.logical_not(np.isnan(diff))]
                        print(len(diff)/len((E_retro - E_descaled)[index]))
                        medians_E_retro.append(np.median(diff))
                        x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
                        x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
                        fe_25 = sum(diff <= diff[x_25])/N
                        fe_75 = sum(diff <= diff[x_75])/N
                        errors_retro.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25**2 + 1/fe_75**2))*(1/1.349))
                        if( k == 0):
                            width_retro = np.array(-diff[x_25]+ diff[x_75])/1.349
                        else:
                            width_retro = np.r_[width_retro,np.array(-diff[x_25]+ diff[x_75])/1.349]
                    
            fig=plt.figure()
            host2 = plt.subplot2grid((4,3),(3,0),colspan =  3)
            rel_imp = ((np.array(width_retro)-np.array(width))/np.array(width))
            host2.plot(means_E,rel_imp,'o')
            host2.yaxis.set_ticks(np.arange(0,1.10,0.20))
            host2.grid()
            host2.set_xlabel(true_cols[c],size = 20)
            host2.set_ylabel('Rel. Imp.', size = 20)
            host2.plot(means_E,np.repeat(0,len(means_E)),color = 'black')
            host = plt.subplot2grid((4,3),(0,0),colspan=3,rowspan=3,sharex = host2)
            par1 = host.twinx()
            host.set_ylabel('W(log(E_pred/E))', size = 20)
            par1.set_ylabel('Count',size = 20)
            n, bins, patches = par1.hist(E_descaled, num_bins, facecolor='blue', alpha=0.1,label = None)            
            host.errorbar(means_E,list(width),errors,linestyle='dotted',fmt = 'o',capsize = 10,label = 'dynedge')
            host.errorbar(means_E,list(width_retro),errors_retro,linestyle='dotted',fmt = 'o',capsize = 10,label = 'retro')
            host.yaxis.set_ticks(np.arange(0,1.1,0.1))
            host.legend(['dynedge GNN','retro'])
            plt.title(' W(log(E_pred/E)) vs E', size = 20)
            host.grid()
            pdf_nam = file
            figs.append(fig)
pdf = matplotlib.backends.backend_pdf.PdfPages("J:\\speciale\\results\\runs\\results\\event_only\\reports\\%s.pdf"%pdf_nam)
for fig in figs: 
    pdf.savefig( fig )
pdf.close()


