from sqlalchemy import create_engine
import sqlalchemy
import pandas as pd
import numpy as np
import argparse
import os
import sqlite3
import time
from sklearn.preprocessing import RobustScaler
import pickle
start_time = time.time()

def fetch_temps(path):
    out = []
    files = os.listdir(path)
    for file in files:
        if '.db' in file:
            out.append(file)
    return out


parser = argparse.ArgumentParser()
parser.add_argument("-path", "--path", type=str, required=True)
parser.add_argument("-outdir", "--outdir", type=str, required=True)
parser.add_argument("-db_name", "--db_name", type=str, required=True)
parser.add_argument("-mode","--mode", type = str, required = True)
args = parser.parse_args()

mode = args.mode
os.makedirs(args.outdir,exist_ok = True)
#scalers = pd.read_pickle('/groups/hep/pcs557/i3_workspace/scalers/dev_classification_000/meta/transformers.pkl')

db_files = fetch_temps(args.path)

print('Found %s .db-files in %s'%(len(db_files),args.path))

#
#
if mode == 'mc-retro':
    print('Making empty %s_unscaled '%args.db_name)
    conn = sqlite3.connect(args.outdir + '/%s_unscaled.db'%(args.db_name))
    c = conn.cursor()
    c.executescript('''
        PRAGMA foreign_keys=off;

        BEGIN TRANSACTION;
        /*create a new table with the same column names and types while
        defining a primary key for the desired column*/
        CREATE TABLE truth (event_no INT PRIMARY KEY NOT NULL,
                                energy_log10 FLOAT NOT NULL,
                                position_x FLOAT NOT NULL,
                                position_y FLOAT NOT NULL,
                                position_z FLOAT NOT NULL,
                                azimuth FLOAT NOT NULL,
                                zenith FLOAT NOT NULL,
                                pid INTEGER NOT NULL,
                                event_time FLOAT NOT NULL,
                                sim_type  NOT NULL,
                                azimuth_retro FLOAT NOT NULL,
                                time_retro FLOAT NOT NULL,
                                energy_log10_retro FLOAT NOT NULL,
                                position_x_retro FLOAT NOT NULL,
                                position_y_retro FLOAT NOT NULL,
                                position_z_retro FLOAT NOT NULL,
                                zenith_retro FLOAT NOT NULL,
                                azimuth_sigma FLOAT NOT NULL,
                                position_x_sigma FLOAT NOT NULL,
                                position_y_sigma FLOAT NOT NULL,
                                position_z_sigma FLOAT NOT NULL,
                                time_sigma FLOAT NOT NULL,
                                zenith_sigma FLOAT NOT NULL,
                                energy_log10_sigma FLOAT NOT NULL,
                                osc_weight FLOAT NOT NULL,
                                interaction_type INTEGER NOT NULL);    
        COMMIT TRANSACTION;

        PRAGMA foreign_keys=on;''')
    c.close()
    conn.close()
if mode == 'data':
    print('Making empty %s_unscaled '%args.db_name)
    conn = sqlite3.connect(args.outdir + '/%s_unscaled.db'%(args.db_name))
    c = conn.cursor()
    c.executescript('''
        PRAGMA foreign_keys=off;

        BEGIN TRANSACTION;
        /*create a new table with the same column names and types while
        defining a primary key for the desired column*/
        CREATE TABLE truth (event_no INT PRIMARY KEY NOT NULL,
                                azimuth_retro FLOAT NOT NULL,
                                time_retro FLOAT NOT NULL,
                                energy_log10_retro FLOAT NOT NULL,
                                position_x_retro FLOAT NOT NULL,
                                position_y_retro FLOAT NOT NULL,
                                position_z_retro FLOAT NOT NULL,
                                zenith_retro FLOAT NOT NULL,
                                azimuth_sigma FLOAT NOT NULL,
                                position_x_sigma FLOAT NOT NULL,
                                position_y_sigma FLOAT NOT NULL,
                                position_z_sigma FLOAT NOT NULL,
                                time_sigma FLOAT NOT NULL,
                                zenith_sigma FLOAT NOT NULL,
                                energy_log10_sigma FLOAT NOT NULL);    
        COMMIT TRANSACTION;

        PRAGMA foreign_keys=on;''')
    c.close()
    conn.close()

print('Empty Truth Created')
conn = sqlite3.connect(args.outdir + '/%s_unscaled.db'%(args.db_name))
c = conn.cursor()
c.executescript('''
    PRAGMA foreign_keys=off;

    BEGIN TRANSACTION;
    /*create a new table with the same column names and types while
    defining a primary key for the desired column*/
    CREATE TABLE features (event_no NOT NULL,
                            charge_log10 FLOAT NOT NULL,
                            dom_time FLOAT NOT NULL,
                            dom_x FLOAT NOT NULL,
                            dom_y FLOAT NOT NULL,
                            dom_z FLOAT NOT NULL,
                            width FLOAT NOT NULL, 
                            pmt_area FLOAT NOT NULL,
                            rqe FLOAT NOT NULL);    
    COMMIT TRANSACTION;
    PRAGMA foreign_keys=on;''')
c.close()
conn.close()
print('Empty Features created')
conn = sqlite3.connect(args.outdir + '/%s_unscaled.db'%(args.db_name))
c = conn.cursor()
c.executescript('''
    PRAGMA foreign_keys=off;
    BEGIN TRANSACTION;
    /*create a new table with the same column names and types while
    defining a primary key for the desired column*/
    CREATE INDEX event_no ON features (event_no);     
    COMMIT TRANSACTION;
    PRAGMA foreign_keys=on;''')
c.close()
conn.close()
print('Index in features created')
#{'charge_log10': charge, 'dom_time': time, 'dom_x': x, 'dom_y': y, 'dom_z': x}
#
#engine_main = sqlalchemy.create_engine('sqlite:///'+args.outdir + '/%s.db'%(args.db_name))
#engine_main.dispose()  

# Creating Unscaled Database
file_counter = 1
for file in db_files:
    print('Extracting %s ( %s / %s)'%(file,file_counter, len(db_files)))
    with sqlite3.connect(args.path + '/' + file) as con:
        query_truth   = 'select * from truth'
        query_feature = 'select * from features'
        feature = pd.read_sql(query_feature,con)
        truth = pd.read_sql(query_truth,con)

    print('Submitting %s to %s'%(file, args.db_name + '_unscaled'))
    engine_main = sqlalchemy.create_engine('sqlite:///'+ args.outdir + '/%s_unscaled.db'%(args.db_name))
    truth.to_sql('truth',engine_main,index= False, if_exists = 'append')
    feature.to_sql('features',engine_main,index= False, if_exists = 'append')
    engine_main.dispose()  

    file_counter += 1
    truth_variables = truth.columns
    feature_variables = feature.columns
 
# Creating Scaled Database
print('Making scaled empty %s '%args.db_name)
conn = sqlite3.connect(args.outdir + '/%s.db'%(args.db_name))
c = conn.cursor()
if mode == 'mc-retro':
    c.executescript('''
        PRAGMA foreign_keys=off;

        BEGIN TRANSACTION;
        /*create a new table with the same column names and types while
        defining a primary key for the desired column*/
        CREATE TABLE truth (event_no INT PRIMARY KEY NOT NULL,
                                energy_log10 FLOAT NOT NULL,
                                position_x FLOAT NOT NULL,
                                position_y FLOAT NOT NULL,
                                position_z FLOAT NOT NULL,
                                azimuth FLOAT NOT NULL,
                                zenith FLOAT NOT NULL,
                                pid INTEGER NOT NULL,
                                event_time FLOAT NOT NULL,
                                sim_type  NOT NULL,
                                azimuth_retro FLOAT NOT NULL,
                                time_retro FLOAT NOT NULL,
                                energy_log10_retro FLOAT NOT NULL,
                                position_x_retro FLOAT NOT NULL,
                                position_y_retro FLOAT NOT NULL,
                                position_z_retro FLOAT NOT NULL,
                                zenith_retro FLOAT NOT NULL,
                                azimuth_sigma FLOAT NOT NULL,
                                position_x_sigma FLOAT NOT NULL,
                                position_y_sigma FLOAT NOT NULL,
                                position_z_sigma FLOAT NOT NULL,
                                time_sigma FLOAT NOT NULL,
                                zenith_sigma FLOAT NOT NULL,
                                energy_log10_sigma FLOAT NOT NULL,
                                osc_weight FLOAT NOT NULL,
                                interaction_type INTEGER NOT NULL);    
        COMMIT TRANSACTION;

        PRAGMA foreign_keys=on;''')
if mode == 'data':
    c.executescript('''
        PRAGMA foreign_keys=off;

        BEGIN TRANSACTION;
        /*create a new table with the same column names and types while
        defining a primary key for the desired column*/
        CREATE TABLE truth (event_no INT PRIMARY KEY NOT NULL,
                                azimuth_retro FLOAT NOT NULL,
                                time_retro FLOAT NOT NULL,
                                energy_log10_retro FLOAT NOT NULL,
                                position_x_retro FLOAT NOT NULL,
                                position_y_retro FLOAT NOT NULL,
                                position_z_retro FLOAT NOT NULL,
                                zenith_retro FLOAT NOT NULL,
                                azimuth_sigma FLOAT NOT NULL,
                                position_x_sigma FLOAT NOT NULL,
                                position_y_sigma FLOAT NOT NULL,
                                position_z_sigma FLOAT NOT NULL,
                                time_sigma FLOAT NOT NULL,
                                zenith_sigma FLOAT NOT NULL,
                                energy_log10_sigma FLOAT NOT NULL);    
        COMMIT TRANSACTION;

        PRAGMA foreign_keys=on;''')

c.close()
conn.close()
print('Empty Truth Created')
conn = sqlite3.connect(args.outdir + '/%s.db'%(args.db_name))
c = conn.cursor()
c.executescript('''
    PRAGMA foreign_keys=off;

    BEGIN TRANSACTION;
    /*create a new table with the same column names and types while
    defining a primary key for the desired column*/
    CREATE TABLE features (event_no NOT NULL,
                            charge_log10 FLOAT NOT NULL,
                            dom_time FLOAT NOT NULL,
                            dom_x FLOAT NOT NULL,
                            dom_y FLOAT NOT NULL,
                            dom_z FLOAT NOT NULL,
                            width FLOAT NOT NULL, 
                            pmt_area FLOAT NOT NULL,
                            rqe FLOAT NOT NULL);   
    COMMIT TRANSACTION;
    PRAGMA foreign_keys=on;''')
c.close()
conn.close()
print('Empty Features created')
conn = sqlite3.connect(args.outdir + '/%s.db'%(args.db_name))
c = conn.cursor()
c.executescript('''
    PRAGMA foreign_keys=off;
    BEGIN TRANSACTION;
    /*create a new table with the same column names and types while
    defining a primary key for the desired column*/
    CREATE INDEX event_no ON features (event_no);     
    COMMIT TRANSACTION;
    PRAGMA foreign_keys=on;''')
c.close()
conn.close()
print('Index in features created')

no_scale = ['event_no','pid','event_time','sim_type', 'interaction_type','osc_weight' ]

truth_scalers = {}
feat_scalers = {}

print(truth_variables)
print(feature_variables)
for var in truth_variables:
    if var not in no_scale and 'retro' not in var and 'sigma' not in var:
        with sqlite3.connect(args.outdir + '/%s_unscaled.db'%(args.db_name)) as con:
            query_truth   = 'select %s from truth'%var
            truth_var = pd.read_sql(query_truth,con)
        print('fitting %s'%var)
        new_scaler = RobustScaler()
        new_scaler.fit(np.array(truth_var[var]).reshape(-1,1))
        truth_scalers[var] = new_scaler


for key in feature_variables:
    if key not in no_scale:
        print('fitting %s'%key)
        with sqlite3.connect(args.outdir + '/%s_unscaled.db'%(args.db_name)) as con:
            query_truth   = 'select %s from features'%key
            feature_var = pd.read_sql(query_truth,con)
        new_scaler = RobustScaler()
        new_scaler.fit(np.array(feature_var[key]).reshape(-1,1))
        feat_scalers[key] = new_scaler

comb_scalers = {'truth':truth_scalers, 'features': feat_scalers}
print('Saving Scalers')
os.makedirs(args.outdir + '/meta',exist_ok= True)
with open(args.outdir + '/meta/transformers.pkl','wb') as handle:
    pickle.dump(comb_scalers,handle,protocol = pickle.HIGHEST_PROTOCOL)

print('Extracing temporaries for last time')


#print('THIS VERSION IS USING FEATURE SCALERS FROM MC DATA BASE')

#truth_scalers = pd.read_pickle('/groups/hep/pcs557/databases/dev_level7_mu_tau_e_retro_000/data/meta/transformers.pkl')['truth']
#feat_scalers = pd.read_pickle('/groups/hep/pcs557/databases/dev_level7_mu_tau_e_retro_000/data/meta/transformers.pkl')['features']

file_counter = 1
for file in db_files:
    print('Extracting %s ( %s / %s)'%(file,file_counter, len(db_files)))
    with sqlite3.connect(args.path + '/' + file) as con:
        query_truth   = 'select * from truth'
        query_feature = 'select * from features'
        feature = pd.read_sql(query_feature,con)
        truth = pd.read_sql(query_truth,con)

    for key in truth_scalers.keys():
        print('transforming %s'%key)
        truth[key] = truth_scalers[key].transform(np.array(truth[key]).reshape(-1,1))
    for key in feat_scalers.keys():
        print('transforming %s'%key)
        feature[key] = feat_scalers[key].transform(np.array(feature[key]).reshape(-1,1))

    print('Submitting %s to %s'%(file, args.db_name))
    engine_main = sqlalchemy.create_engine('sqlite:///'+ args.outdir + '/%s.db'%(args.db_name))
    truth.to_sql('truth',engine_main,index= False, if_exists = 'append')
    feature.to_sql('features',engine_main,index= False, if_exists = 'append')
    engine_main.dispose()  

    file_counter += 1


print('Done! Time elapsed: %s min'%((time.time() - start_time)/60))
