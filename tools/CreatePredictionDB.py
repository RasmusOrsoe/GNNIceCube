import pandas as pd
import os
import sqlalchemy
import sqlite3

def GetPredictions(path):
    models  = os.listdir(path)
    results = []
    for model in models:
        results.append(pd.read_csv(path + '\\' + model + '\\' + 'results.csv'))
    
    return models,results

def CreatePredictionDB(handle):
    
    path_pred = r'X:\speciale\results\%s'%handle
    path = r'X:\speciale\data\graphs\%s\predictions'%handle
    os.makedirs(path,exist_ok = True)
    
    
    
    models,results = GetPredictions(path_pred)
    
    for k in range(0,len(models)):
        if '-' in models[k]:
            models[k] = str('_').join(models[k].split('-'))
            
    engine_main = sqlalchemy.create_engine('sqlite:///' + path + '\\%s'%'ensemble_predictions.db')
    for j in range(0,len(models)):
        result = results[j]
        model  = models[j]
        print('INSERTING %s'%model)
        data_batch = result.loc[:,['event_no','E_pred','azimuth_pred','zenith_pred']]
        data_batch.to_sql(model,engine_main,index= False, if_exists = 'append',chunksize = len(data_batch))
    truth = pd.DataFrame(data_batch['event_no'].unique())
    truth.columns = ['event_no']
    truth.to_csv(path + '\\events.csv')


handle  = 'dev_numu_train_l5_retro_001\event_only_aggr_test_SRT_energy_scaled_new_weights_for_submodels'

CreatePredictionDB(handle)