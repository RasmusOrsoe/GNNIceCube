import sqlite3
import pandas as pd

db_file = r'X:\aml\MetObs_test12.db'

with sqlite3.connect(db_file) as con:
    query = 'select parameterId from variable_meta where update_frequency = 10'
    variables = pd.read_sql(query,con)
    
    
with sqlite3.connect(db_file) as con:
    query = 'select time from %s'%variables['parameterId'][0]
    unique_time = pd.unique(pd.read_sql(query,con)['time'])
    
    
test_keys = variables['parameterId'][0:5]

k = 0
for key in test_keys:
    with sqlite3.connect(db_file) as con:
        query = 'select * from %s where time in %s'%(key, str(tuple(unique_time['time'])))
        data = pd.read_sql(query,con).sort_values('time').reset_index(drop = True)
    if k == 0:
        res = data
        k = k + 1
    else:
        res = pd.concat([res,data], axis = 1, ignore_index = True)
    
