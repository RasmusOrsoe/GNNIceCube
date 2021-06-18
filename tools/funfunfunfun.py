import pandas as pd
import sqlite3 


sets = r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\meta\sets.pkl'

sets = pd.read_pickle(sets)

db_file = r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\data\dev_lvl7_mu_nu_e_classification_v003.db'


events = sets['train']['event_no']
with sqlite3.connect(db_file) as con:
    query  = 'select event_no from truth where event_no in %s and abs(pid) != 13'%(str(tuple(events)))
    train_events = pd.read_sql(query,con)
    
events = sets['test']['event_no']
with sqlite3.connect(db_file) as con:
    query  = 'select event_no from truth where event_no in %s and abs(pid) != 13'%(str(tuple(events)))
    test_events = pd.read_sql(query,con)
    
    
train_events.to_csv(r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\selections\neutrino_train.csv')
test_events.to_csv(r'X:\speciale\data\raw\dev_lvl7_mu_nu_e_classification_v003\selections\neutrino_test.csv')