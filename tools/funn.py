import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sqlite3
import sqlalchemy


path = r'X:\speciale\results\dev_upgrade_train_step4_001\event_only_SRT_upgrade_martin_selection_5inputs\dynedge-E-protov2-zenith\results.csv'

scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_upgrade_train_step4_001\meta\transformers.pkl')

results =  pd.read_csv(path)

zenith =np.squeeze(scalers['truth']['zenith'].inverse_transform(np.array(results['zenith']).reshape(-1,1)))
zenith_pred = np.squeeze(scalers['truth']['zenith'].inverse_transform(np.array(results['zenith_pred']).reshape(-1,1)))

fig, ax = plt.subplots()
my_cmap = plt.cm.jet
h = plt.hist2d(zenith,zenith_pred,bins=50,range=[[0,3],[0,3]],cmap = my_cmap)
plt.xlabel('$zenith_{true}$ [Rad.]', size = 20)
plt.ylabel('$zenith_{prediction}$ [Rad.]',size = 20)
plt.title('Zenith regression on cut upgrade data',size = 15)
plt.plot(np.arange(0,3,0.01),np.arange(0,3,0.01),color = 'red')
fig.colorbar(h[3], ax=ax)

####
#%%
db_path = r'X:\speciale\data\raw\dev_classification_corsika_000\data\dev_classification_corsika_000.db'

db_path2 = r'X:\speciale\data\raw\dev_classification_000\data\dev_classification_000.db'

with sqlite3.connect(db_path) as con:
    query = 'select event_no from truth' 
    dat = pd.read_sql(query, con)
    
events = dat.loc[:,'event_no']
events = events[0:10]    
    
with sqlite3.connect(db_path) as con:
    query = 'select * from truth where event_no in %s'%(str(tuple(events))) 
    truth = pd.read_sql(query, con)
    query = 'select event_no from features' 
    features = pd.read_sql(query, con)
 #%%   



    
rows = np.arange(0,len(features),1)
rows = pd.DataFrame(rows)   
rows.columns = ['row'] 
engine = sqlalchemy.create_engine('sqlite:///' + db_path)

rows.to_sql('features',engine,index= False, if_exists = 'append')
engine.dispose()
 
#%%
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.executescript('''
    PRAGMA foreign_keys=off;

    BEGIN TRANSACTION;
    ALTER TABLE truth RENAME TO old_table;

    /*create a new table with the same column names and types while
    defining a primary key for the desired column*/
    CREATE TABLE truth (event_no INT PRIMARY KEY NOT NULL,
                            position_x FLOAT,
                            position_y FLOAT,
                            position_z FLOAT,
                            zenith FLOAT,
                            azimuth FLOAT,
                            energy_log10 FLOAT,
                            pid INTEGER);

    INSERT INTO truth SELECT * FROM old_table;

    DROP TABLE old_table;
    
    
    COMMIT TRANSACTION;

    PRAGMA foreign_keys=on;''')

#close out the connection
c.close()
conn.close()