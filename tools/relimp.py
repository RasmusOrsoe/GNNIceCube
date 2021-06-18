import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def grabfiles(path):
    files = os.listdir(path)
    res = []
    files_out = []
    for file in files:
        if '.csv' in file:
            files_out.append(file)
            res.append(pd.read_csv(path + '\\' + file))
        
    return res, files_out
        
def rel_imp_error(path):
    data, files = grabfiles(path)
    warehouse = {}
    keys = ['energy', 'zenith']
    for key in keys:
        warehouse[key] = []
    for k in range(0,len(data)):
        for key in keys:
            if key in files[k]:
                warehouse[key].append(data[k])
    plot_this = {}
    for key in keys:
        k = 0
        for element in range(0,len(warehouse[key])):
            if k == 0:
                plot_this[key] = warehouse[key][element]
                k = k+1
            else:
                plot_this[key]['precut'] = (plot_this[key]['precut'] + warehouse[key][element]['precut'])/2
                plot_this[key]['error'] = (plot_this[key]['precut'] - warehouse[key][element]['precut'])
        
    

    fig = plt.figure()
    k = 0
    for key in keys:
        variable = key
        entry = plot_this[key]
        plt.errorbar(entry['E'], entry['precut'],entry['error'],marker='o', markeredgecolor='black',label = variable)
        print('%s : %s'%(variable, np.mean(entry['precut'][0:3])))
        print(entry['error'])
        #if 'cut' in data[k].columns:
        #    plt.plot(data[k]['E'], data[k]['cut'], marker='D', markeredgecolor='black',label = variable + ' 50% Most Certain')
        #    print('%s : %s'%(variable, np.mean(data[k]['cut'][0:3])))    
        k +=1
    plt.plot(np.repeat(0,5), color = 'black', lw = 3)    
    plt.legend(fontsize = 20)
    plt.ylabel('Rel. Improvement', size = 30)
    plt.xlabel('$Energy_{log_{10}}$', size = 30)
    plt.title('Relative Improvement', size = 30)
    plt.grid()
    plt.yticks(np.arange(-0.2,0.35,0.05),fontsize = 20)
    plt.xticks(fontsize = 30)
    fig.tight_layout() 

path = r'X:\speciale\litterature\conclusion\check'

rel_imp_error(path)
#%%
path = r'X:\speciale\litterature\conclusion\check'

data,files = grabfiles(path)

fig = plt.figure()
k = 0
for file in files:
    variable = file.split('.')[0]
    plt.plot(data[k]['E'], data[k]['precut'], marker='D', markeredgecolor='black',label = variable)
    print('%s : %s'%(variable, np.mean(data[k]['precut'][0:3])))
    if 'cut' in data[k].columns:
        plt.plot(data[k]['E'], data[k]['cut'], marker='D', markeredgecolor='black',label = variable + ' 50% Most Certain')
        print('%s : %s'%(variable, np.mean(data[k]['cut'][0:3])))    
    k +=1
plt.plot(np.repeat(0,5), color = 'black', lw = 3)    
plt.legend(fontsize = 20)
plt.ylabel('Rel. Improvement', size = 30)
plt.xlabel('$Energy_{log_{10}}$', size = 30)
plt.title('Relative Improvement', size = 30)
plt.grid()
plt.yticks(np.arange(-0.2,0.35,0.05),fontsize = 20)
plt.xticks(fontsize = 30)
fig.tight_layout()