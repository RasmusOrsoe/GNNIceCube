import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from collections import Counter

res = pd.read_csv('J:\\speciale\\results\\counts\\counts.csv')
res.columns = ['event_no','count','type']

type0 = res['count'][res['type'] == 0]
plt.figure()
for typ in [0,1,2]:
    type0 = res['count'][res['type'] == typ]
    n, bins, patches = plt.hist(x=type0, bins='auto',
                                alpha=0.7, rwidth=0.85)
plt.xlim([0,200])
plt.legend(['0','1','2'])
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Nodes')
plt.ylabel('Frequency')
plt.title('Node Size Frequency')


event_no_0 = list()
event_no_1 = list()
event_no_2 = list()

for k in range(0,len(res)):
    if res['type'][k] == 0:
        if res['count'][k] > 40 and res['count'][k] < 48:
            event_no_0.append(res['event_no'][k])
    if res['type'][k] == 1:
        if res['count'][k] > 40 and res['count'][k] < 48:
            event_no_1.append(res['event_no'][k])
    if res['type'][k] == 2:
        if res['count'][k] > 40 and res['count'][k] < 48:
            event_no_2.append(res['event_no'][k])



db_file_list =  ['C:\\applied_ML\\final_project\\data\\120000_00.db',
            'C:\\applied_ML\\final_project\\data\\140000_00.db',
            'C:\\applied_ML\\final_project\\data\\160000_00.db']                    #
scalar = pd.DataFrame()                                                             #
sequential = pd.DataFrame()                                                         #
for db_file in db_file_list:                                                        #
    with sqlite3.connect(db_file) as con:                                           #
        query = 'select * from sequential'                                          # MERGES ALL .db FILES TO TWO .csv FILES:
        sequential = sequential.append(pd.read_sql(query, con))                     # scalar.csv , sequential.csv   
        query = 'select * from scalar'                                              # THESE ARE THEN WRITTEN TO DRIVE
        scalar = scalar.append(pd.read_sql(query, con))                             #
        cursor = con.cursor()                                                       #
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        
scalar = scalar.reset_index().drop(columns = ['level_0', 'index'])
sequential = sequential.reset_index().drop(columns = ['level_0', 'index'])



type0_seq = pd.DataFrame()
type0_sca = pd.DataFrame()
type1_seq = pd.DataFrame()
type1_sca = pd.DataFrame()
type2_seq = pd.DataFrame()
type2_sca = pd.DataFrame()
for k in range(0,len(event_no_0[0:20000])):
    index_seq = sequential['event_no'] == event_no_0[k]
    index_sca = scalar['event_no'] == event_no_0[k]
    type0_seq = type0_seq.append(sequential.loc[index_seq,:])
    type0_sca = type0_sca.append(scalar.loc[index_sca,:])
    
    index_seq = sequential['event_no'] == event_no_1[k]
    index_sca = scalar['event_no'] == event_no_1[k]
    type1_seq = type1_seq.append(sequential.loc[index_seq,:])
    type1_sca = type1_sca.append(scalar.loc[index_sca,:])
    
    index_seq = sequential['event_no'] == event_no_2[k]
    index_sca = scalar['event_no'] == event_no_2[k]
    type2_seq = type2_seq.append(sequential.loc[index_seq,:])
    type2_sca = type2_sca.append(scalar.loc[index_sca,:])

    print('%s/%s'%(k+1,20000))
    
type0_seq.to_csv(r'J:\\speciale\\results\\40-48-node-split\\type0_seq.csv',index=False)
type0_sca.to_csv(r'J:\\speciale\\results\\40-48-node-split\\type0_sca.csv',index=False)
type1_seq.to_csv(r'J:\\speciale\\results\\40-48-node-split\\type1_seq.csv',index=False)
type1_sca.to_csv(r'J:\\speciale\\results\\40-48-node-split\\type1_sca.csv',index=False)
type2_seq.to_csv(r'J:\\speciale\\results\\40-48-node-split\\type2_seq.csv',index=False)
type2_sca.to_csv(r'J:\\speciale\\results\\40-48-node-split\\type2_sca.csv',index=False)    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    