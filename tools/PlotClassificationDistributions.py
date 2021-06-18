import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from matplotlib import gridspec

def fix_output(output):
    output['out2'] = expit(output['out2'])
    output['out1'] = expit(output['out1'])
    output['weight'] = np.repeat(1/len(output),len(output))
    return output

def MakePlots(mc,data, mode, name):
    if mode == 0:
        fig, ax = plt.subplots(2,1)
        #ax = fig.get_axes()
        ax[0].hist(mc['out1'],label = 'mc muon', color = 'blue', bins = 50,weights = mc['weight'])
        ax[0].hist(mc['out2'],label = 'mc $\\nu$ score', color = 'green', bins = 50,alpha = 0.8,weights = mc['weight'])
        ax[0].hist(mc['out2'][mc['mc'] == 1],label = 'mc $\\nu$', color = 'orange', bins = 50,alpha = 0.8,weights = mc['weight'][mc['mc'] == 1], histtype = 'step')
        ax[0].hist(mc['out2'][mc['mc'] == 0],label = 'mc $\\mu$', color = 'yellow', bins = 50,alpha = 0.8,weights = mc['weight'][mc['mc'] == 0], histtype = 'step')
        #plt.hist(data['out1'],label = 'data muon',histtype = 'step', color = 'red', bins = 50,density = True)
        ax[0].hist(data['out2'],label = 'data $\\nu$ score',histtype = 'step', color = 'black', bins = 50,weights = data['weight'])
        ax[0].legend()
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Normalized Log Count', size = 20)
        plt.title(name, size = 20)
        #ax[0].xticks(fontsize=15)
        #ax[0].yticks(fontsize=15)
    if mode == 1:
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        #fig, ax = plt.subplots(2,1)
        #plt.hist(mc['out1'],label = 'mc muon', color = 'blue', bins = 50,density = True)
        N, bins, _ = ax0.hist(mc['out2'],label = 'mc', color = 'green', bins = 50,alpha = 0.8,weights = mc['weight'])
        ax0.hist(mc['out2'][mc['mc'] == 1],label = 'mc $\\nu$', color = 'red', bins = 50,alpha = 0.8,weights = mc['weight'][mc['mc'] == 1], histtype = 'step')
        ax0.hist(mc['out2'][mc['mc'] == 0],label = 'mc $\\mu$', color = 'blue', bins = 50,alpha = 0.8,weights = mc['weight'][mc['mc'] == 0], histtype = 'step')
        #plt.hist(data['out1'],label = 'data muon',histtype = 'step', color = 'red', bins = 50,density = True)
        N_data,_ , _ = ax0.hist(data['out2'],label = 'data',histtype = 'step', color = 'black', bins = 50,weights = data['weight'])
        ax0.legend(fontsize=15)
        ax0.set_yscale('log')
        ax0.set_ylabel('Normalized Log Count', size = 20)
        ax1.set_xlabel('$\\nu$ Score', size = 20)
        #plt.xticks(fontsize=15)
        #plt.yticks(fontsize=15)
        ax0.set_xticks([])
        ax0.set_title(name, size = 20)
        print(len(bins))
        #ax1.errorbar(x=bins[0:50],y=N_data/N, yerr=(N_data/N - np.sqrt(N)), linestyle = 'dotted',fmt = 'x',capsize = 10)
        linear_model=np.polyfit(bins[0:50],N_data/N,1)
        linear_model_fn=np.poly1d(linear_model)
        chi2 = ((linear_model_fn(bins[0:50]) - N_data/N)/(N_data/N)).sum()
        ax1.plot(bins[0:50], N_data/N, 'x')
        plt.plot(bins[0:50], linear_model_fn(bins[0:50]), label = 'a = -0.71 \n b = 1.54 \n $\\chi^2$ = %s'%round(chi2,2))
        print(chi2)
        ax1.set_ylabel('data/mc', size = 20)
        ax1.plot(np.arange(0,1, 0.001), np.repeat(1,1000), color = 'black')
        ax1.legend(fontsize = 15)
        ax1.tick_params(axis = 'x',labelsize = 15)
        print(np.sqrt(N))
    #fig.tight_layout()
    return

mc = pd.read_csv(r'X:\speciale\results\dev_level7_mu_tau_e_muongun_classification\classification_realistic_shuffled\dynedge-protov2-classification-k=8-c3not3-classification\results.csv')
data = pd.read_csv(r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler_realistic\classification\dynedge-protov2-classification-k=8-c3not3-classification-realistic\results.csv')

mc = fix_output(mc)
data = fix_output(data)

#MakePlots(mc,data, 1, 'realistic model')
#MakePlots(mc,data, 0, 'realistic model')


mc = pd.read_csv(r'X:\speciale\results\dev_level7_mu_tau_e_muongun_classification\classification_even_sample\dynedge-protov2-classification-k=8-c3not3-classification\results.csv')
data = pd.read_csv(r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler\burnsample\dynedge-protov2-classification-k=8-c3not3-classification\results.csv')

mc = fix_output(mc)
data = fix_output(data)

MakePlots(mc,data,1, 'lvl7 $\\nu / \\mu $ Classification (1% sample)')
#MakePlots(mc,data, 0, '$\\nu / \\mu $ Classification')



