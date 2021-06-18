import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def grabdata(key,scale):
    data_db = r'X:\speciale\data\raw\oscnext_IC8611_newfeats_000_mc_scaler\data\oscnext_IC8611_newfeats_000_mc_scaler.db'
    
    with sqlite3.connect(data_db) as con:
        query = 'select %s from features'%key
        data = pd.read_sql(query,con)
    if scale:
        scaler = pd.read_pickle(r'X:\speciale\data\raw\oscnext_IC8611_newfeats_000_mc_scaler\meta\transformers.pkl')['features'][key]
        data[key] = scaler.inverse_transform(np.array(data[key]).reshape(-1,1))
    return data

def grabmc(key,events,scale,pid):
    mc_db = r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\data\dev_level7_mu_e_tau_oscweight_newfeats.db'
    scaler = pd.read_pickle(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\meta\transformers.pkl')['features'][key]
    if pid != None:
        with sqlite3.connect(mc_db) as con:
            query = 'select event_no from truth where event_no in %s and abs(pid) = %s'%(str(tuple(events)),pid)
            pid_events = pd.read_sql(query,con)
        pid_events = pid_events['event_no']
    else:
        pid_events = events

    with sqlite3.connect(mc_db) as con:
        query = 'select %s from features where event_no in %s'%(key,str(tuple(pid_events)))
        data = pd.read_sql(query,con)
        
    if scale:
        data[key] = scaler.inverse_transform(np.array(data[key]).reshape(-1,1))
        
    return data





pids = None #[12,14,16]
events = pd.read_csv(r'X:\speciale\data\raw\dev_level7_mu_e_tau_oscweight_newfeats\event_selection\mc_data_classification.csv')['event_no']
keys = ['dom_x', 'dom_y', 'dom_z']
scale = False

for key in keys:
    if scale:
        if key == 'dom_x':
            bins = np.arange(-100,200, 40)
        if key == 'dom_y':
            bins = np.arange(-200,150, 40)
        if key == 'dom_z':
            bins = np.arange(-500,-300,40)
    else:
        bins= np.arange(-5,5,0.1)
    
    if pids != None:
        fig = plt.figure()
        data = grabdata(key,scale)
        plt.hist(data[key],bins = bins, label = 'data',density = True) 
        for pid in pids:
            mc = grabmc(key, events, scale, pid)
            plt.hist(mc[key],bins = bins, histtype = 'step', label = 'mc ' + str(pid), density = True)
    
    if pids == None:
        fig = plt.figure()
        data = grabdata(key,scale)
        plt.hist(data[key],bins = bins, label = 'data',density = True)
        mc = grabmc(key, events, scale, pids)
        plt.hist(mc[key],bins = bins, histtype = 'step', label = 'mc ', density = True)
        
    plt.xlabel(key, size = 20)
    plt.ylabel('Density', size = 20)
    plt.title('mc-data: %s'%key,size = 30)
    plt.legend()
        