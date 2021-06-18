import pandas as pd
import numpy as np
path = r'X:\speciale\hep\arrays_from_hep\arrays\SRTInIcePulses'
data = np.load(path + '\data.npy')
index = np.load(path + '\index.npy')
truth = np.load(r'X:\speciale\hep\arrays_from_hep\arrays\MCInIcePrimary\data.npy')

#%%
import pickle
lol = pickle.load(open(r'X:\\speciale\\hep\\arrays_from_hep\\resources\\icu_gcd.p', 'rb'), encoding='latin1')