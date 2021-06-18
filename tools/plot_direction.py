import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

def GetResults(path):
    model_names = os.listdir(path)
    results = list()
    for name in model_names:
        if name != 'skip':
            print(name)
            model_res = pd.read_csv(path + "\\%s\\results.csv"%name)
            results.append(model_res)
    
    return model_names,results




path = r'X:\speciale\results\dev_numu_train_l5_retro_001\event_only_aggr_test'

models,results = GetResults(path)
figs = list()
error_list = list()
means_list = list()

radian_to_degree = 360/(2*np.pi)

scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\meta\transformers.pkl')

azimuth_scaler = scalers['truth']['zenith']

result = results[0]
result = result.sort_values('event_no').reset_index(drop=True)
result = result.sort_values('E')
num_bins = 10
E = np.array(result.loc[:,'E']).reshape(-1,1)
n, bins, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
plt.close()
for result in results:
    azimuth_median = list()
    azimuth_error = list()
    means_E = list()
    azimuth_pred = azimuth_scaler.inverse_transform(np.array(result.loc[:,'zenith_pred']).reshape(-1,1))*radian_to_degree
    azimuth = azimuth_scaler.inverse_transform(np.array(result.loc[:,'zenith']).reshape(-1,1))*radian_to_degree
    for k in range(len(bins)-1):
        index_pred = (azimuth_pred >= bins[k]) & (azimuth_pred <= bins[k+1])
        index = (E >= bins[k]) & (E <= bins[k+1])
        azimuth_median.append(np.median(azimuth[index] - azimuth_pred[index]))
        means_E.append(np.mean(E[index]))
        
        diff = (-azimuth + azimuth_pred)[index]
        x_25 = np.where(diff == np.percentile(diff,25,interpolation='nearest'))[0] #int(0.16*N)
        x_75 = np.where(diff == np.percentile(diff,75,interpolation='nearest'))[0]
        if( k == 0):
            azimuth_error = np.array([np.median(diff) - diff[x_25], 
                               np.median(diff) - diff[x_75]])
        else:
            azimuth_error = np.c_[azimuth_error,np.array([np.median(diff) - diff[x_25],
                                            np.median(diff) - diff[x_75]])]
    
    #azimuth_error = azimuth_error - azimuth_median
    #plt.hist2d(np.squeeze(np.array(azimuth)),np.squeeze(np.array(azimuth_pred)),bins = 50,cmin = 100)
    
    plt.errorbar(means_E,azimuth_median,abs(azimuth_error),linestyle='dotted',fmt = 'o',capsize = 10)
plt.legend(models)
