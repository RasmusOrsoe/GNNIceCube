# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

path  = r'X:\speciale\data\export\dom_counting'

files = os.listdir(path)

for k in range(0,len(files)):
    
    df = pd.read_csv(path + '\\%s'%files[k])
    if k == 0:
        results = df
    else:
        results = results.append(df)
        
results.columns = ['n_doms','energy_log10'] 
        
transformers = pd.read_pickle(r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\meta\transformers.pkl')
scaler = transformers['truth']['energy_log10']

energy = scaler.inverse_transform(np.array(results['energy_log10']).reshape((-1,1)))

plt.hist2d(results['n_doms'],energy.flatten(),bins = 300)
plt.xlim((None,150))
plt.ylabel('Energy_log10 [GeV]',size = 15)
plt.xlabel('Number of Activated DOMs',size  = 15)
plt.title('Energy vs DOM Count',size = 22)