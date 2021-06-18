import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import pickle
#db_file = r'X:\speciale\data\raw\dev_level7_mu_tau_e_muongun_classification\data\dev_level7_mu_tau_e_muongun_classification.db' 
db_file = r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\data\dev_lvl7_mu_nu_e_classification_v003.db'
event_nos = pd.DataFrame()
    
#%%

with sqlite3.connect(db_file) as con:
   query = 'select event_no from truth where abs(pid) = 13 '
   muon_data = pd.read_sql(query,con).sample(frac=1)
   
   query = 'select event_no from truth where abs(pid) = 12 '
   v_e_data = pd.read_sql(query,con).sample(frac=1)
   
   query = 'select event_no from truth where abs(pid) = 14 '
   v_u_data = pd.read_sql(query,con).sample(frac=1)
   
   query = 'select event_no from truth where abs(pid) = 16 '
   v_t_data = pd.read_sql(query,con).sample(frac=1)


#%%

n_muons = len(muon_data)

n_ve = len(v_e_data)
n_vu = len(v_u_data)
n_vt = len(v_t_data)

muon_train, muon_test = train_test_split( muon_data, test_size=0.2, random_state=42)
v_e_train, v_e_test = train_test_split( v_e_data, test_size=0.2, random_state=42)
v_u_train, v_u_test = train_test_split( v_u_data, test_size=0.2, random_state=42)
v_t_train, v_t_test = train_test_split( v_t_data, test_size=0.2, random_state=42)



events_train  = muon_train.append(v_e_train).append(v_u_train).append(v_t_train)
events_test  = muon_test.append(v_e_test).append(v_u_test).append(v_t_test)

#%%

events_train.reset_index(drop = True)
events_test.reset_index(drop = True)

sets = {'train': events_train, 'test': events_test}

#torch.save(sets, r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\meta\sets.pkl')

with open(r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\meta\sets.pkl','wb') as handle:
    pickle.dump(sets,handle,protocol = pickle.HIGHEST_PROTOCOL)