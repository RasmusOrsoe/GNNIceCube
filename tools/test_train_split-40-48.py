import pandas as pd
import os

path = 'J:\\speciale\\results\\40-48-node-split'
path_write = r'J:\\speciale\\results\\40-48-node-split\test-train\\'
x = 0
for file in os.listdir(path):    
    if file.endswith(".csv"):
        if file[5:9] == '_seq':
            types_seq = file[0:5]
            seq = pd.read_csv(os.path.join(path, file)).drop(columns = ['index','level_0'])
        if file[5:9] == '_sca':
            types_sca = file[0:5]
            sca = pd.read_csv(os.path.join(path, file))
    if x != 0:
        if types_seq == types_sca :
            train_event_sca = sca.loc[0:9999,:]
            valid_event_sca = sca.loc[10000:19999,:].reset_index()
            
            train_event_seq = pd.DataFrame()
            valid_event_seq = pd.DataFrame()
            for k in range(0,len(train_event_sca)):
                index =  seq['event_no'] == train_event_sca['event_no'][k]
                train_event_seq = train_event_seq.append(seq.loc[index,:])
                
                index =  seq['event_no'] == valid_event_sca['event_no'][k]
                valid_event_seq = valid_event_seq.append(seq.loc[index,:])
            
            valid_event_sca = valid_event_sca.drop(columns = 'index')
            
            valid_event_sca.to_csv(os.path.join(path_write, 'valid_scalar_%s.csv'%types_seq),index = False)
            
            valid_event_seq.to_csv(os.path.join(path_write, 'valid_sequential_%s.csv'%types_seq),index = False)
            
            train_event_sca.to_csv(os.path.join(path_write, 'train_scalar_%s.csv'%types_seq),index = False)
            
            train_event_seq.to_csv(os.path.join(path_write, 'train_seq_%s.csv'%types_seq),index = False)
            
    x  = x + 1