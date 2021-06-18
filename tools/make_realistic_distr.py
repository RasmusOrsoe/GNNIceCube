import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_file = r'X:\speciale\data\raw\dev_level7_mu_tau_e_muongun_classification\data\dev_level7_mu_tau_e_muongun_classification.db' 

event_nos = pd.DataFrame()
    
    
v_t    = 60*10-6    
muon   = 40*10-6
v_u = 700*10-6
v_e    = 200*10-6    


total = v_t + v_u + v_e

v_t = v_t/total
#muon = muon/total
v_u = v_u/total
v_e = v_e/total
#%%

with sqlite3.connect(db_file) as con:
   query = 'select event_no from truth where abs(pid) = 13 '
   muon_data = pd.read_sql(query,con)
   
   query = 'select event_no from truth where abs(pid) = 12 '
   v_e_data = pd.read_sql(query,con)
   
   query = 'select event_no from truth where abs(pid) = 14 '
   v_u_data = pd.read_sql(query,con)
   
   query = 'select event_no from truth where abs(pid) = 16 '
   v_t_data = pd.read_sql(query,con)
   #query = 'select * from features limit 1'
   #features = pd.read_sql(query,con)

n_muon = len(muon_data)
n_max = n_muon#int(n_muon/(muon*100)*100)

N=int((n_muon*(1-0.151))/(0.151*3))

v_e_max = int(v_t*3*N)#int(v_e*(n_max))
v_t_max = int(v_u*3*N)#int(v_t*n_max)
v_u_max = int(v_e*3*N)#int(v_u*n_max)


 
event_nos = event_nos.append(muon_data, ignore_index = True)
event_nos = event_nos.append(v_e_data.sample(v_e_max).reset_index(drop = True), ignore_index = True)
event_nos = event_nos.append(v_t_data.sample(v_t_max).reset_index(drop = True), ignore_index = True)
event_nos = event_nos.append(v_u_data.sample(v_u_max).reset_index(drop = True), ignore_index = True)

event_nos = event_nos.sample(frac = 1)