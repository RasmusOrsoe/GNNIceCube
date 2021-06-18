import pandas as pd
import yaml
import json
import ast
import datetime
import requests
import urllib.request, json 
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def CalculateVariableCounts(stations, variable_list):
    counts = {}
    freq10_variables = variable_list['parameterId'][variable_list['update frequency'] == '10 min']
    for variable in freq10_variables:
        counts[variable]= [0.0]
    for station in range(0,len(stations)):
        params = stations['parameterId'][station]
        for param in params:
            if param in counts.keys():
                counts[param] = [counts[param][0] + 1]
            
    return pd.DataFrame(counts)

def CleanColumns(data):
    for key in data.columns:
        if sum(data[key] == None) > 0:
            data[key][data[key] == None] = -1
        if sum(pd.isna(data[key])) > 0:
            data[key][pd.isna(data[key])] = -1
        
    return data
    
    
path = r'X:\aml\2000-06.txt'

variable_list = pd.read_csv(r'X:\aml\variable_list.csv')


url_stations = 'https://dmigw.govcloud.dk/v2/metObs/collections/station/items?api-key=4c22176d-a8a1-4dc1-8078-4e2f531ca120'

stations = []
with urllib.request.urlopen(url_stations) as url:
    request_data = json.loads(url.read().decode())
    
n_stations = len(request_data['features'])
for station in range(0,n_stations):
    a = request_data['features'][station]['properties']
    stations.append(a)
stations = CleanColumns(pd.DataFrame(stations))
#%%
station_type = 'Synop'
keys = list(variable_list['parameterId'][variable_list['update frequency'] == '10 min'])
requested_stations = []
for station in range(0,n_stations):
    if stations.loc[station,'type'] == station_type:
        available_parameters = stations.loc[station,'parameterId']
        for available_parameter in available_parameters:
            if (stations['stationId'][station] in requested_stations) == False:
                if available_parameter in keys:
                    requested_stations.append(stations['stationId'][station])
                    
#%%

counts = CalculateVariableCounts(stations, variable_list).T
counts.plot(kind='bar')                
#%%                

api_key = '4c22176d-a8a1-4dc1-8078-4e2f531ca120'
datetime = '2020-01-12T00:00:00Z/2020-01-13T00:00:00Z'
template = {'Latitude':[-1], 'Longitude':[-1], 'time':[-1], 'station':[-1]}
keys = variable_list['parameterId']
variable_data_dict = {}
for key in keys:
    variable_data_dict[key] = 0
    


for key in variable_list['parameterId']:
    is_first = True
    variable_data = deepcopy(template)
    for station in stations['stationId']:
        request = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items?parameterId=%s&stationId=%s&datetime=%s&api-key=%s"%(key,station,datetime,api_key)
        with urllib.request.urlopen(request) as url:
            request_data = json.loads(url.read().decode())
        requests = request_data['features']
        
        for data in requests:
            if is_first:
                temp_df = pd.DataFrame(data['geometry']['coordinates'])
                temp_df = temp_df.T
                temp_df.columns =['Latitude', 'Longitude']
                
                variable_data['time']        = [data['properties']['observed']]
                variable_data['Latitude']    = [temp_df['Latitude'].values[0]]
                variable_data['Longitude']   = [temp_df['Longitude'].values[0]]
                variable_data['station']     = [data['properties']['stationId']]
                variable_data[key]           = [data['properties']['value']]
                is_first = False
            else:
                temp_df = pd.DataFrame(data['geometry']['coordinates'])
                temp_df = temp_df.T
                temp_df.columns =['Latitude', 'Longitude']
                
                variable_data['time'].append(data['properties']['observed'])
                variable_data['Latitude'].append(temp_df['Latitude'].values[0])
                variable_data['Longitude'].append(temp_df['Longitude'].values[0])
                variable_data['station'].append(data['properties']['stationId'])
                variable_data[key].append(data['properties']['value'])

    variable_data_dict[key] = pd.DataFrame(variable_data)
#%%   
from sqlalchemy import create_engine
import sqlite3
def create_index(db_path, key):
    con = sqlite3.connect(db_path)
    c = con.cursor()
    c.executescript("PRAGMA foreign_keys=off;\nBEGIN TRANSACTION;\nCREATE INDEX time ON {} (time);\nCOMMIT TRANSACTION;\nPRAGMA foreign_keys=on;".format(key))
    c.close()
    con.close()
    
def MakeDatabase(db_path, variable_data_dict, station_meta, variable_meta):
    keys = variable_data_dict.keys()
    
    for key in keys:
        con = sqlite3.connect(db_path)
        column_names = variable_data_dict[key].columns.values.tolist()
        k = 0
        for column_name in column_names:
            if column_name in ['Latitude', 'Longitude', 'station', key]:
                if k == 0:
                    query_columns = column_name + ' FLOAT NOT NULL'
                    k +=1
                else:
                    query_columns = query_columns + ', ' + column_name + ' FLOAT NOT NULL'
                    
            if column_name == 'time':
                query_columns = query_columns + ', ' + column_name + ' DATETIME NOT NULL'
                
                
        CODE = "PRAGMA foreign_keys=off;\nCREATE TABLE {} ({});\nPRAGMA foreign_keys=on;".format(key,query_columns)
        c = con.cursor()
        c.executescript(CODE)
        c.close()
        con.close()
        engine = create_engine('sqlite:///' + db_path)
        variable_data_dict[key].to_sql(key, engine,index = False, if_exists = 'append')
        
    #station_meta.to_sql('station_meta', engine, index = False, if_exists =  'append')
    variable_meta.to_sql('variable_meta', engine, index = False, if_exists = 'append')
    
db_path = r'X:\aml\MetObs_test12.db'
MakeDatabase(db_path, variable_data_dict, stations, variable_list)    