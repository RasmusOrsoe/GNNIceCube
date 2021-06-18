import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def grab(path):
    files = os.listdir(path)
    res = pd.DataFrame()
    for file in files:
        res = res.append(pd.read_csv(path + '\\' + file).reset_index(drop = True))
    return res


res = grab(r'X:\speciale\data\export\Upgrade Cleaning\threepmt_martin_selection').reset_index(drop = True)
res.to_csv(r'X:\speciale\data\export\Upgrade Cleaning\selections\rasmus_selection\rasmus_selection.csv')

#real_bad = pd.read_csv(r'X:\speciale\data\export\Upgrade Cleaning\duplicates\bad\events.csv')
#stat_bad = grab(r'X:\speciale\data\export\Upgrade Cleaning\statcount_bad')
#stat_good = grab(r'X:\speciale\data\export\Upgrade Cleaning\statcount_good')
#real_good = pd.read_csv(r'X:\speciale\data\export\Upgrade Cleaning\duplicates\good\events.csv')

#percent_good = sum(stat_good['0'] > 0)/len(real_good)
#percent_bad  = sum(stat_bad['0'] < 0)/len(real_bad)

#bpe_bad = (stat_bad['0']/(stat_bad['1']))

#bpe_good = (stat_good['0']/stat_good['1'])

#print(bpe_bad.mean())
#print(bpe_good.mean())


#fig = plt.figure()
#plt.hist(stat_bad['0'], histtype = 'step',label = 'bad', bins = np.arange(0,100,1), density = True)
#plt.hist(stat_good['0'], histtype = 'step',label = 'good', bins = np.arange(0,100,1), density = True)
#plt.legend()

