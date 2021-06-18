import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy.special import softmax
from scipy.special import expit
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import sqlite3

def CalculateDensity(results):
    
    
    score = results['out2']/(results['out1'] + results['out2'])
    neutrino = sum(results['mc'][score > 0.7] == 1)
    passed_events = sum(score > 0.7)
    neutrino_percent_s = neutrino/passed_events
    notneutrino = sum(results['mc'][score > 0.7] == 0)
    notneutrino_percent_s = notneutrino/passed_events
    
    notneutrino_percent = notneutrino/sum(results['mc']==0)
    neutrino_percent = neutrino/sum(results['mc']==1)
    print('------ stats of passed events -----')
    print('SAMPLE PREDICTED TO: \n \
          Neutrino Percentage : %s \n \
          NotNeutrino Percentage     : %s' %(neutrino_percent_s,notneutrino_percent_s))
    print('------ keep fractions -----')
    print('SAMPLE PREDICTED TO: \n \
          Neutrino Percentage : %s \n \
          NotNeutrino Percentage     : %s' %(neutrino_percent,notneutrino_percent))
    

def SoftMax(x):
    f = np.exp(x - np.max(x))/np.sum(np.exp(x))
    return f

def MakePlots(results):
        #### ID TABLE   ############
    # muon:         0  # 0 # 0 #
    # stopped_muon: 1  # 1 # 0 #
    # v_e         : 2  # 2 # 1 #
    # v_u         : 3  # 2 # 1 #
    # v_t         : 4  # 2 # 1 #
    
    pred  = []
    truth = []
    data_acc         = 0
    muon_stopped_acc = 0
    mc_acc            = 0
    
    p_count = 0
    # check = results[['out1','out2']]
    # for j in range(0,len(results)):
    #     pid = results.loc[j,'mc']
        
    #     if check.loc[j,'out2'] > check.loc[j,'out1']:
    #         current_pred = 1
    #     else:
    #         current_pred = 0
        
    #     pred.append(current_pred)
    #     truth.append(pid)
    #     if pid == 0 and current_pred == 0:
    #         data_acc += 1
    #     #if pid == 1 and current_pred == 1:
    #     #    muon_stopped_acc += 1
    #     if pid == 1 and current_pred == 1:
    #         mc_acc += 1
    #     if p_count == 1000:
    #         print('%s / %s'%(j, len(results)))
    #         p_count = 0
    #     p_count = p_count + 1
        
        
    # data_acc            =   data_acc/sum(np.array(truth) == 0)
    
    # mc_acc               =   mc_acc/sum(np.array(truth) == 1) 
    # print('data accuracy: %s'%data_acc)
    # #print('stopped muon accuracy: %s'%muon_stopped_acc)
    # print('mc accuracy: %s'%mc_acc)
    
    thresholds = np.arange(0,1,0.001)
    
    tpr = []
    fpr = []
    i = 1
    for threshold in thresholds:
        print( '%s / %s '%(i,len(thresholds)))
        index   = results['out2'] >= threshold
        index2  = results['out2'] <= threshold
        tp    = sum(results['mc'][index] == 1)/(sum(results['mc'][index] == 1) +sum(results['mc'][index2] == 1) )
        if  ((sum(results['mc'][index] == 0) +sum(results['mc'][index2] == 0)) != 0): 
             fp    = sum(results['mc'][index] == 0)/(sum(results['mc'][index] == 0) +sum(results['mc'][index2] == 0) )
        else:
            fp = np.nan
        tpr.append(tp)
        fpr.append(fp)
        i += 1
    

     
    return fpr,tpr
    #plt.plot(fpr,tpr)

def CalculateTroelsThreshold(results):
    thresholds = np.arange(0,1,0.001)
    
    tpr = []
    fpr = []
    i = 1
    for threshold in thresholds:
        print( '%s / %s '%(i,len(thresholds)))
        
        
        scores = results['out2']/(results['out2'] + results['out1'])
        
        index   = scores >= threshold
        index2  = scores <= threshold
        
        tp    = sum(results['mc'][index] == 1)/(sum(results['mc'][index] == 1) +sum(results['mc'][index2] == 1) )

        fp    = sum(results['mc'][index] == 0)/(sum(results['mc'][index] == 0) +sum(results['mc'][index2] == 0) )

        tpr.append(tp)
        fpr.append(fp)
        i += 1
    return fpr,tpr

def CalculateMPS(results):
    tpr = []
    fpr = []
    i = 1

    #print( '%s / %s '%(i,len(thresholds)))
    
    
    scores = results['out2']/(results['out2'] + results['out1'])
    
    index   = results['out2'] > results['out1']
    
    index2  = ~index
    
    tp    = sum(results['mc'][index] == 1)/(sum(results['mc'][index] == 1) +sum(results['mc'][index2] == 1) )

    fp    = sum(results['mc'][index] == 0)/(sum(results['mc'][index] == 0) +sum(results['mc'][index2] == 0) )

    tpr.append(tp)
    fpr.append(fp)
    i += 1

    return tpr, fpr
def CalculatePoint(threshold,results):
    tpr = []
    fpr = []
    i = 1

    #print( '%s / %s '%(i,len(thresholds)))
    
    
   
    
    index   = results['out2'] >= threshold
    
    index2  = ~index
    
    tp    = sum(results['mc'][index] == 1)/(sum(results['mc'][index] == 1) +sum(results['mc'][index2] == 1) )

    fp    = sum(results['mc'][index] == 0)/(sum(results['mc'][index] == 0) +sum(results['mc'][index2] == 0) )

    tpr.append(tp)
    fpr.append(fp)

    return tpr, fpr

path = r'x:\speciale\results\dev_level7_mu_e_tau_oscweight_000\for_mc_data_classification\dynedge-protov2-classification-k=8-c3not3-test\results.csv'
path2 = r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\for_mc_data_classification\dynedge-protov2-classification-k=8-c3not3-test\results.csv'
#path3 = r'X:\speciale\results\dev_level7_mu_tau_e_muongun_classification\classification_realistic_subsample\dynedge-protov2-classification-k=8-c3not3-classification\results.csv'
#path4 = r'X:\speciale\data\graphs\dev_level7_mu_tau_e_muongun_classification\classification_realistic_subsample_test\dynedge-protov2-classification-k=8-c3not3-classification\results.csv'
#path5 = r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler_realistic\classification_realistic_subsample_v2\dynedge-protov2-classification-k=8-c3not3-classification\results.csv'
path6 = r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler_realistic\classification_realistic_subsample_v3\dynedge-protov2-classification-k=8-c3not3-classification\results.csv'


#path7 = r'X:\speciale\results\dev_level7_mu_tau_e_muongun_classification\classification_even_sample\dynedge-protov2-classification-k=8-c3not3-classification\results.csv'

#path8 = r'X:\speciale\results\dev_level7_mu_tau_e_muongun_classification\classification_36percentmuon_realistic_shuffled\dynedge-protov2-classification-k=8-c3not3-classification\results.csv'

#path9 = r'X:\speciale\results\dev_level7_mu_tau_e_muongun_classification\classification_realistic_shuffled\dynedge-protov2-classification-k=8-c3not3-classification\results.csv'

#path10 = r'X:\speciale\results\dev_level7_mu_tau_e_muongun_classification\classification_151percentmuon_realistic\dynedge-protov2-classification-k=8-c3not3-classification-realistic\results.csv'
path_ct = r'X:\speciale\results\dev_level7_mu_e_tau_oscweight_newfeats\track_cascadev2\dynedge-protov2-classification-k=8-c3not3-track_cascade_good\results.csv'

path11 = r'X:\speciale\data\export\classification\gnn+lightgbm\results.csv'
path_lvl4 = r'X:\speciale\results\dev_level4_muon_noise_nu\classification\dynedge-protov2-classification-k=8-c3not3-w_test_val\results.csv'
retro_mc_lvl4 = r'X:\speciale\data\raw\dev_level4_muon_noise_nu\data\dev_level4_muon_noise_nu.db'
scalers_lvl4 = pd.read_pickle(r'X:\speciale\data\raw\dev_level4_muon_noise_nu\meta\transformers.pkl' )
path12 = r'X:\speciale\results\dev_level2_mu_tau_e_muongun_classification_wnoise\3mio_even_noise_muon_neutrino_multilabel_classifier\dynedge-protov2-classification-k=8-c3not3-classification_multilabel\results.csv'
retro_mc = r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\data\dev_lvl7_mu_nu_e_classification_v003.db'
scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\meta\transformers.pkl')
with sqlite3.connect(retro_mc) as con:
    query = 'select pid, lvl7_probnu from truth'
    retro = pd.read_sql(query,con)

retro['lvl7_probnu'] = scalers['truth']['lvl7_probnu'].inverse_transform(np.array(retro['lvl7_probnu']).reshape(-1,1))
retro['pid'][(abs(retro['pid']) == 12) | (abs(retro['pid']) == 14) | (abs(retro['pid']) == 16)] = 1
retro['pid'][(abs(retro['pid']) == 13)]  = 0
retro.columns = ['mc', 'out2']
retro = retro.sample(frac = 1)

with sqlite3.connect(retro_mc_lvl4) as con:
    query = 'select pid, lvl4_probnu from truth'
    retro_lvl4 = pd.read_sql(query,con)
retro_lvl4['lvl4_probnu'] = scalers_lvl4['truth']['lvl4_probnu'].inverse_transform(np.array(retro_lvl4['lvl4_probnu']).reshape(-1,1))
retro_lvl4['pid'][(abs(retro_lvl4['pid']) == 1)]  = 0
retro_lvl4['pid'][(abs(retro_lvl4['pid']) == 12) | (abs(retro_lvl4['pid']) == 14) | (abs(retro_lvl4['pid']) == 16)] = 1
retro_lvl4['pid'][(abs(retro_lvl4['pid']) == 13)]  = 0

retro_lvl4.columns = ['mc', 'out2']
retro_lvl4 = retro_lvl4.sample(frac = 1)

#%%

#### lvl3
fpr_retro,tpr_retro, _ = roc_curve(retro['mc'], retro['out2'])


lvl3 = pd.read_csv(r'X:\speciale\results\dev_level2_mu_tau_e_muongun_classification_wnoise\3mio_even_noise_classifier\dynedge-protov2-classification-k=8-c3not3-classification\results.csv')
lvl3.columns = ['Unnamed: 0', 'event_no', 'out1', 'out2', 'mc']
lvl3['out1'] = expit(lvl3['out1'])
lvl3['out2'] = expit(lvl3['out2'])

fpr_lvl3, tpr_lvl3, _ = roc_curve(lvl3['mc'], lvl3['out2']/(lvl3['out1'] + lvl3['out2']))

lvl3_neutrino = pd.read_csv(r'X:\speciale\results\dev_level2_mu_tau_e_muongun_classification_wnoise\3mio_even_noise_muonisalsonoise_classifier\dynedge-protov2-classification-k=8-c3not3-classification_muonisnoise\results.csv')
lvl3_neutrino.columns = ['Unnamed: 0', 'event_no', 'out1', 'out2', 'mc']
lvl3_neutrino['out1'] = expit(lvl3_neutrino['out1'])
lvl3_neutrino['out2'] = expit(lvl3_neutrino['out2'])

fpr_lvl3_neutrino, tpr_lvl3_neutrino,_ = roc_curve(lvl3_neutrino['mc'], lvl3_neutrino['out2']/(lvl3_neutrino['out1'] + lvl3_neutrino['out2']))

lvl3_multilabel = pd.read_csv(path12)
lvl3_multilabel['out1'] = expit(lvl3_multilabel['out1'])
lvl3_multilabel['out2'] = expit(lvl3_multilabel['out2'])
lvl3_multilabel['out3'] = expit(lvl3_multilabel['out3'])

fpr_multi, tpr_multi, _ = roc_curve(lvl3_multilabel['pid']==2, lvl3_multilabel['out3'])#/(lvl3_multilabel['out1'] + lvl3_multilabel['out2'] + lvl3_multilabel['out3']))

CalculateDensity(lvl3_neutrino)
##### lvl4
fpr_retro_lvl4,tpr_retro_lvl4, _ = roc_curve(retro_lvl4['mc'], retro_lvl4['out2'])
lvl4 = pd.read_csv(path_lvl4)
lvl4.columns = ['Unnamed: 0', 'event_no', 'out1', 'out2', 'mc']
lvl4['out1'] = expit(lvl4['out1'])
lvl4['out2'] = expit(lvl4['out2'])

fpr_lvl4, tpr_lvl4,_ = roc_curve(lvl4['mc'], lvl4['out2']/(lvl4['out1'] + lvl4['out2']))

## track cascade
lvl7_tc = pd.read_csv(path_ct)
lvl7_tc.columns = ['Unnamed: 0', 'event_no', 'out1', 'out2', 'mc']
lvl7_tc['out1'] = expit(lvl7_tc['out1'])
lvl7_tc['out2'] = expit(lvl7_tc['out2'])

fpr_lvltc, tpr_lvltc,_ = roc_curve(lvl7_tc['mc'],lvl7_tc['out2']/(lvl7_tc['out1'] + lvl7_tc['out2']))


### lvl7
results = pd.read_csv(path11)
#results['out1'] = expit(results['out1'])
#results['out2'] = expit(results['out2'])
#results = CalculateDensity(results)

a = results.loc[:,['out1','out2','mc']]
b = results.loc[:,['prob0','prob1','mc']]
b.columns = ['out1','out2','mc']
#fpr_a,tpr_a = MakePlots(a)
#fpr_t, tpr_t = CalculateTroelsThreshold(a)

fpr_sklearn, tpr_sklearn, _ = roc_curve(a['mc'], a['out2']/(a['out1'] + a['out2']))
#fpr_b,tpr_b = MakePlots(b)
#results = pd.read_csv(path7)
#results['out1'] = expit(results['out1'])
#results['out2'] = expit(results['out2'])
#results = CalculateDensity(results)
#fpr_7,tpr_7,fpr_72,tpr_72 = MakePlots(results)



#results = pd.read_csv(path)
#results['out1'] = expit(results['out1'])
#results['out2'] = expit(results['out2'])

#fpr_0,tpr_0 = MakePlots(results)

#results = pd.read_csv(path2)
#results['out1'] = expit(results['out1'])
#results['out2'] = expit(results['out2'])

#fpr_1,tpr_1 = MakePlots(results)

#results = pd.read_csv(path6)
#results['out1'] = expit(results['out1'])
#results['out2'] = expit(results['out2'])

#fpr_5,tpr_5 = MakePlots(results)

#results = CalculateDensity(results)

#%%
#plt.plot(fpr_0,tpr_0,label = '$ \\nu_{\mu,\\tau, e} $ : Individual Scalers')
#plt.plot(fpr_1,tpr_1, label = '$\\nu_{\mu,\\tau, e} $ : MC Scaler')
#
#plt.plot(fpr_7,tpr_7, label = 'muon-neutrino classification 50-50')

#plt.plot(fpr_9,tpr_9, label = 'muon-neutrino classification realistic')
#plt.plot(fpr_72,tpr_72, label = 'muon-neutrino classification 50-50 v2')
#plt.plot(fpr_92,tpr_92, label = 'muon-neutrino classification realistic v2')


#plt.plot(fpr_b,tpr_b, label = 'muon-neutrino classification 50-50 lightgbm')
#plt.plot(fpr_3,tpr_3, label = 'MC Scaler, realistic2')
#plt.plot(fpr_4,tpr_4, label = 'MC Scaler class., realistic')

#plt.plot(fpr_5,tpr_5, label = '$\\nu_{\mu,\\tau, e}$ and $\mu_{muongun}$ : MC Scaler')

fig = plt.figure()
#plt.plot(fpr_lvl3, tpr_lvl3, label = 'noise classifier')
plt.plot(fpr_lvl3_neutrino, tpr_lvl3_neutrino, label = 'neutrino classifier')
#plt.plot(fpr_multi, tpr_multi, label = 'multilabel classifier')
plt.xlabel('FP', size = 40)
plt.ylabel('TP', size = 40)
auc_score = auc(fpr_lvl3,tpr_lvl3)
auc_score_neutrino = auc(fpr_lvl3_neutrino,tpr_lvl3_neutrino)
#auc_score_multi    = auc(fpr_multi, tpr_multi)
col_labels=['AUC']
row_labels=['noise classifier',
            'neutrino classifier']
table_vals=[[round(auc_score,3)],
            [round(auc_score_neutrino,3)]]

the_table = plt.table(cellText=table_vals,
                  colWidths = [0.1]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center right')

#y,x = CalculatePoint(0.7,lvl3)
y_neutrino,x_neutrino = CalculatePoint(0.7,lvl3_neutrino)
plt.text(x_neutrino[0] + 0.01,y_neutrino[0], '(%s,%s)'%(str(round(x_neutrino[0],2)), str(round(y_neutrino[0],2))))
#plt.plot(x,y, 'o')
plt.plot(x_neutrino,y_neutrino, 'o')
plt.plot(np.repeat(x_neutrino, 1000),np.arange(0,1,0.001), '--', label = 'OscNext Selection')
plt.legend(fontsize = 20)

#%%
fig = plt.figure()
plt.plot(fpr_retro,tpr_retro, label = 'OscNext "lvl7_probnu"')

#plt.plot(fpr_8,tpr_8, label = '$\\nu_{\mu,\\tau, e}$ and $\mu_{muongun} 36\%$ : MC Scaler')
#plt.plot(fpr_10,tpr_10, label = '$\\nu_{\mu,\\tau, e}$ and $\mu_{muongun} 15.1\%$ : MC Scaler')

plt.xlabel('False Positive Rate', size = 40)
plt.ylabel('True Positive Rate', size = 40)
auc_score = auc(fpr_sklearn,tpr_sklearn)
auc_score_retro = auc(fpr_retro,tpr_retro)
col_labels=['AUC']
row_labels=['$dynedge$',
            '"lvl7_probnu"']
table_vals=[[round(auc_score,3)],
            [round(auc_score_retro,3)]]
#col_labels=['col1','col2','col3']
#row_labels=['row1','row2','row3']
#table_vals=[[11,12,13],
#            [21,22,23],
#            [31,32,33]]
# the rectangle is where I want to place the table
the_table = plt.table(cellText=table_vals,
                  colWidths = [0.1]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center right')



#plt.text(0.5,0.5,table, size = 20, color = 'red')
#plt.text(0.6,0.45, '(via sklearn.metrics.auc)', size  = 25, color = 'red')
#plt.plot(np.repeat(0.040058070760296806, len(np.arange(0,0.9298044141396017,0.0001))),np.arange(0,0.9298044141396017,0.0001), '--', color = 'black')
#plt.plot(np.arange(0,0.040058070760296806, 0.0001), np.repeat(0.9298044141396017, len(np.arange(0,0.040058070760296806, 0.0001))), '--', color = 'black')
#plt.plot(0.040058070760296806,0.9298044141396017, 'o', label = '$dynedge$ MPS')
#plt.text(0.040058070760296806 - 0.06,0.9298044141396017 + 0.05, '(0.040,0.93)', size  = 15)
y,x = CalculateMPS(a)

y_retro, x_retro = CalculatePoint(0.7, retro)
plt.plot(fpr_sklearn, tpr_sklearn, label = 'dynedge')
plt.text(x_retro[0] + 0.02, 0.1, 'OscNext Selection', rotation = 'vertical', color = 'orange', fontsize = 15)
plt.plot(x_retro, tpr_sklearn[ np.argmin(abs(fpr_sklearn - x_retro))], 'o')
plt.plot( fpr_sklearn[ np.argmin(abs(tpr_sklearn - y_retro))], y_retro, 'o')
#plt.text(fpr_sklearn[np.argmin(abs(tpr_sklearn - y_retro))] - 0.06,y_retro[0], '(%s,%s)'%(str(round(fpr_sklearn[ np.argmin(abs(tpr_sklearn - y_retro))],2)), str(round(y_retro[0],2))))
plt.plot(x,y,'o', label = 'dynedge MPS')
plt.text(x_retro[0] + 0.01,y_retro[0], '(%s,%s)'%(str(round(x_retro[0],2)), str(round(y_retro[0],2))))
plt.text(x[0] - 0.03,y[0] + 0.03, '(%s,%s)'%(str(round(x[0],2)), str(round(y[0],2))))
plt.text(x_retro[0] + 0.01, tpr_sklearn[ np.argmin(abs(fpr_sklearn - x_retro))], '(%s,%s)'%(str(round(x_retro[0],2)), str(round( tpr_sklearn[ np.argmin(abs(fpr_sklearn - x_retro))],2))))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.repeat(x_retro, 1000),np.arange(0,1,0.001), '--')
plt.title('level7 oscNext MC GNN ROC Curve', size = 30)

plt.annotate(
     '(%s,%s)'%(str(round(fpr_sklearn[ np.argmin(abs(tpr_sklearn - y_retro))],2)), str(round(y_retro[0],2))),
    xy=( fpr_sklearn[ np.argmin(abs(tpr_sklearn - y_retro))], y_retro[0]), xycoords='data',
    xytext=(80, -50), textcoords='offset points',
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3,rad=-0.2"))

plt.plot(x_retro,y_retro,'o')
#plt.title('MC - Measurement Classification',size = 40)
#plt.plot(fpr_a,tpr_a, label = '$dynedge$')
#plt.plot(fpr_t,tpr_t, label = '$dynedge_t$')


plt.legend(fontsize = 20)
#%%
fig = plt.figure()
plt.plot(fpr_retro_lvl4,tpr_retro_lvl4, label = 'OscNext "lvl4_probnu"')

plt.xlabel('False Positive Rate', size = 40)
plt.ylabel('True Positive Rate', size = 40)
auc_score = auc(fpr_lvl4,tpr_lvl4)
auc_score_retro = auc(fpr_retro_lvl4,tpr_retro_lvl4)
col_labels=['AUC']
row_labels=['$dynedge$',
            '"lvl4_probnu"']
table_vals=[[round(auc_score,3)],
            [round(auc_score_retro,3)]]
#col_labels=['col1','col2','col3']
#row_labels=['row1','row2','row3']
#table_vals=[[11,12,13],
#            [21,22,23],
#            [31,32,33]]
# the rectangle is where I want to place the table
the_table = plt.table(cellText=table_vals,
                  colWidths = [0.1]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center right')



#plt.text(0.5,0.5,table, size = 20, color = 'red')
#plt.text(0.6,0.45, '(via sklearn.metrics.auc)', size  = 25, color = 'red')
#plt.plot(np.repeat(0.040058070760296806, len(np.arange(0,0.9298044141396017,0.0001))),np.arange(0,0.9298044141396017,0.0001), '--', color = 'black')
#plt.plot(np.arange(0,0.040058070760296806, 0.0001), np.repeat(0.9298044141396017, len(np.arange(0,0.040058070760296806, 0.0001))), '--', color = 'black')
#plt.plot(0.040058070760296806,0.9298044141396017, 'o', label = '$dynedge$ MPS')
#plt.text(0.040058070760296806 - 0.06,0.9298044141396017 + 0.05, '(0.040,0.93)', size  = 15)
y,x = CalculateMPS(lvl4)

y_retro, x_retro = CalculatePoint(0.7, retro_lvl4)
plt.plot(fpr_lvl4, tpr_lvl4, label = 'dynedge')
plt.text(x_retro[0] + 0.02, 0.1, 'OscNext Selection', rotation = 'vertical', color = 'orange', fontsize = 15)
plt.plot(x_retro, tpr_lvl4[ np.argmin(abs(fpr_lvl4 - x_retro))], 'o')
plt.plot( fpr_lvl4[ np.argmin(abs(tpr_lvl4 - y_retro))], y_retro, 'o')
#plt.text(fpr_lvl4[np.argmin(abs(tpr_lvl4 - y_retro))] - 0.06,y_retro[0], '(%s,%s)'%(str(round(fpr_lvl4[ np.argmin(abs(tpr_lvl4 - y_retro))],2)), str(round(y_retro[0],2))))
plt.plot(x,y,'o', label = 'dynedge MPS')
plt.text(x_retro[0] + 0.01,y_retro[0], '(%s,%s)'%(str(round(x_retro[0],2)), str(round(y_retro[0],2))))
plt.text(x[0] - 0.1,y[0] + 0.03, '(%s,%s)'%(str(round(x[0],2)), str(round(y[0],2))))
plt.text(x_retro[0] + 0.01, tpr_lvl4[ np.argmin(abs(fpr_lvl4 - x_retro))], '(%s,%s)'%(str(round(x_retro[0],2)), str(round( tpr_lvl4[ np.argmin(abs(fpr_lvl4 - x_retro))],2))))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.repeat(x_retro, 1000),np.arange(0,1,0.001), '--')
plt.title('level4 oscNext MC GNN ROC Curve', size = 30)

plt.annotate(
     '(%s,%s)'%(str(round(fpr_lvl4[ np.argmin(abs(tpr_lvl4 - y_retro))],2)), str(round(y_retro[0],2))),
    xy=( fpr_lvl4[ np.argmin(abs(tpr_lvl4 - y_retro))], y_retro[0]), xycoords='data',
    xytext=(80, -50), textcoords='offset points',
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3,rad=-0.2"))

plt.plot(x_retro,y_retro,'o')
#plt.title('MC - Measurement Classification',size = 40)
#plt.plot(fpr_a,tpr_a, label = '$dynedge$')
#plt.plot(fpr_t,tpr_t, label = '$dynedge_t$')


plt.legend(fontsize = 20)
    
#%%
## track cascade
tc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'
with sqlite3.connect(tc_db) as con:
    query  = 'select event_no, energy_log10 from truth where event_no in %s'%(str(tuple()))


fig = plt.figure()
#plt.plot(fpr_retro_lvl4,tpr_retro_lvl4, label = 'OscNext "lvl4_probnu"')

plt.xlabel('False Positive Rate', size = 40)
plt.ylabel('True Positive Rate', size = 40)
auc_score = auc(fpr_lvltc,tpr_lvltc)
#auc_score_retro = auc(fpr_retro_lvl4,tpr_retro_lvl4)
col_labels=['AUC']
row_labels=['$dynedge$']
table_vals=[[round(auc_score,3)]]
#col_labels=['col1','col2','col3']
#row_labels=['row1','row2','row3']
#table_vals=[[11,12,13],
#            [21,22,23],
#            [31,32,33]]
# the rectangle is where I want to place the table
the_table = plt.table(cellText=table_vals,
                  colWidths = [0.1]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center right')



#plt.text(0.5,0.5,table, size = 20, color = 'red')
#plt.text(0.6,0.45, '(via sklearn.metrics.auc)', size  = 25, color = 'red')
#plt.plot(np.repeat(0.040058070760296806, len(np.arange(0,0.9298044141396017,0.0001))),np.arange(0,0.9298044141396017,0.0001), '--', color = 'black')
#plt.plot(np.arange(0,0.040058070760296806, 0.0001), np.repeat(0.9298044141396017, len(np.arange(0,0.040058070760296806, 0.0001))), '--', color = 'black')
#plt.plot(0.040058070760296806,0.9298044141396017, 'o', label = '$dynedge$ MPS')
#plt.text(0.040058070760296806 - 0.06,0.9298044141396017 + 0.05, '(0.040,0.93)', size  = 15)
#y,x = CalculateMPS(lvl4)

y, x = CalculatePoint(0.7, lvl7_tc)
plt.plot(fpr_lvltc, tpr_lvltc, label = 'dynedge')
#plt.text(x_retro[0] + 0.02, 0.1, 'OscNext Selection', rotation = 'vertical', color = 'orange', fontsize = 15)
#plt.plot(x_retro, tpr_lvl4[ np.argmin(abs(fpr_lvl4 - x_retro))], 'o')
#plt.plot( fpr_lvl4[ np.argmin(abs(tpr_lvl4 - y_retro))], y_retro, 'o')
#plt.text(fpr_lvl4[np.argmin(abs(tpr_lvl4 - y_retro))] - 0.06,y_retro[0], '(%s,%s)'%(str(round(fpr_lvl4[ np.argmin(abs(tpr_lvl4 - y_retro))],2)), str(round(y_retro[0],2))))
plt.plot(x,y,'o', label = '0.7 threshold')
#plt.text(x_retro[0] + 0.01,y_retro[0], '(%s,%s)'%(str(round(x_retro[0],2)), str(round(y_retro[0],2))))
#plt.text(x[0] - 0.1,y[0] + 0.03, '(%s,%s)'%(str(round(x[0],2)), str(round(y[0],2))))
plt.text(x[0] + 0.01,y[0], '(%s,%s)'%(str(round(x[0],2)), str(round( y[0],2))))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.plot(np.repeat(x_retro, 1000),np.arange(0,1,0.001), '--')
plt.title('level7 oscNext MC Cascade/Track ROC Curve', size = 30)

#plt.annotate(
#     '(%s,%s)'%(str(round(fpr_lvl4[ np.argmin(abs(tpr_lvl4 - y_retro))],2)), str(round(y_retro[0],2))),
#    xy=( fpr_lvl4[ np.argmin(abs(tpr_lvl4 - y_retro))], y_retro[0]), xycoords='data',
#    xytext=(80, -50), textcoords='offset points',
#    arrowprops=dict(arrowstyle="->",
#                    connectionstyle="arc3,rad=-0.2"))

#plt.plot(x_retro,y_retro,'o')
#plt.title('MC - Measurement Classification',size = 40)
#plt.plot(fpr_a,tpr_a, label = '$dynedge$')
#plt.plot(fpr_t,tpr_t, label = '$dynedge_t$')


plt.legend(fontsize = 20)
    
        
#%%
### real data

path = r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler_realistic\classification\dynedge-protov2-classification-k=8-c3not3-classification\results.csv'
path2 = r'X:\speciale\results\oscnext_IC8611_newfeats_000_mc_scaler_realistic\classification\dynedge-protov2-classification-k=8-c3not3-classification-realistic\results.csv'
path3 = r'X:\speciale\results\oscnext_level7_IC8611_for-thesis\classification\dynedge-protov2-classification-k=8-c3not3-classification-realistic\results.csv'

results = CalculateDensity(path)
results2 = CalculateDensity(path2)
results3 = CalculateDensity(path3)        
