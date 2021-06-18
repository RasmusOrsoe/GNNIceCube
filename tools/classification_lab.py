import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import lightgbm as ltb
from scipy.special import expit
from sklearn.model_selection import train_test_split

def calculate_metrics(results, keys):
    p_count  = 0
    neutrino = 0
    muon     = 0
    fpfp = 0
    tptp = 0
    results = endout
    check = results[[keys[0],keys[1],'mc']]
    pred  = []
    for j in range(0,len(results)):
        if check.loc[j,keys[1]] > check.loc[j,keys[0]]:
            current_pred = 1
        else:
            current_pred = 0
            
        if current_pred == 0 and check.loc[j,'mc'] == 0:
            
            muon  += 1
        #current_pred == 1:
        #    muon_stopped_acc += 1
        if current_pred == 1 and check.loc[j,'mc'] == 1:
            neutrino +=1
        
        if current_pred == 1 and check.loc[j,'mc'] == 0:
            fpfp +=1
        if p_count == 1000:
            print('%s / %s'%(j, len(results)))
            p_count = 0
        p_count = p_count + 1
    
    neutrino_percent = sum(check['mc'] == 1)/len(check)
    muon_percent     = sum(check['mc'] == 0)/len(check)
    
    neutrino_accuracy = neutrino/sum(check['mc'] == 1)
    muon_accuracy = muon/sum(check['mc'] == 0)
    
    muinnu =sum((results[keys[1]] > results[keys[0]]) & (results['mc'] == 0))/  sum((results[keys[1]] > results[keys[0]]))
    
    print('SAMPLE PREDICTED TO: \n \
          Neutrino Percentage : %s \n \
          Neutrino Accuracy   : %s \n \
          Muon Percentage     : %s \n \
          Muon Accuracy       : %s \n \
          Percent Muons in nu : %s' %(neutrino_percent,neutrino_accuracy,muon_percent, muon_accuracy, muinnu))
    
    print(fpfp/len(results))
    print(tptp/len(results))
    

data = pd.read_csv(r'X:\speciale\results\dev_level7_mu_tau_e_muongun_classification\classification_even_sample\dynedge-protov2-classification-k=8-c3not3-classification\results.csv')

data = data.sample(frac = 1)

data['out1'] = expit(data['out1'])
data['out2'] = expit(data['out2'])

data_list = np.array_split(data,10)

x_train, x_test, y_train, y_test = train_test_split(data.loc[:,['out1','out2']], data['mc'], test_size=0.30)


clf = ltb.LGBMClassifier()

clf.fit(x_train, y_train)


endout = pd.DataFrame()

out = clf.predict_proba(x_test)
out = pd.DataFrame(out)
out.columns = ['prob0', 'prob1']
results = pd.DataFrame(out)


endout = pd.concat([pd.DataFrame(x_test).reset_index(drop = True), pd.DataFrame(y_test).reset_index(drop = True),results.reset_index(drop = True)], axis  = 1, ignore_index= True)

endout.columns = ['out1', 'out2', 'mc', 'prob0','prob1']
endout = endout.reset_index(drop = True)

calculate_metrics(endout, ['out1','out2'])
print('---------LIGHT GBM --------')
calculate_metrics(endout,['prob0','prob1'])
#%%
p_count  = 0
neutrino = 0
muon     = 0
results = endout
check = results[['out1','out2','mc']]
pred  = []
for j in range(0,len(results)):
    if check.loc[j,'out2'] > check.loc[j,'out1']:
        current_pred = 1
    else:
        current_pred = 0
        
    if current_pred == 0 and check.loc[j,'mc'] == 0:
        muon  += 1
    #current_pred == 1:
    #    muon_stopped_acc += 1
    if current_pred == 1 and check.loc[j,'mc'] == 1:
        neutrino += 1
    if p_count == 1000:
        print('%s / %s'%(j, len(results)))
        p_count = 0
    p_count = p_count + 1

neutrino_percent = sum(check['mc'] == 1)/len(check)
muon_percent     = sum(check['mc'] == 0)/len(check)

neutrino_accuracy = neutrino/sum(check['mc'] == 1)
muon_accuracy = muon/sum(check['mc'] == 0)

muinnu =sum((results['out2'] > results['out1']) & (results['mc'] == 0))/  sum((results['out2'] > results['out1']))

print('SAMPLE PREDICTED TO: \n \
      Neutrino Percentage : %s \n \
      Neutrino Accuracy   : %s \n \
      Muon Percentage     : %s \n \
      Muon Accuracy       : %s \n \
      Percent Muons in nu : %s' %(neutrino_percent,neutrino_accuracy,muon_percent, muon_accuracy, muinnu))
#%%
results.to_csv(r'X:\speciale\data\export\classification\gnn+lightgbm\results.csv')