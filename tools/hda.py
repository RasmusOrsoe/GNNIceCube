import time 
from torch_geometric.data import Data 
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from copy  import deepcopy
from multiprocessing import Pool
 
def MakeGraph(seq,sca,node_size,path,file, SRT = False, edge_config = 'Time', scaling = 'Standard', fullgrid = False): 
    if scaling == 'Standard':
        scaler = StandardScaler()
    now = time.time() 
    graphs = list() 
    if SRT == False and edge_config == 'Time' and fullgrid == False :
        for event in range(0,len(sca['event_no'])): 
            index = seq['event_no'] == sca['event_no'][event] 
            x = seq.loc[index,['dom_x',
                               'dom_y',
                               'dom_z',
                               'dom_charge',
                               'dom_time']].sort_values('dom_time') 
            if len(x) < node_size: 
                diff = abs(len(x) - node_size) 
                for i in range(0,diff): 
                    x.append(pd.DataFrame(np.zeros([1,5]), columns = x.columns)).reset_index(drop = True) 
            if len(x) > node_size:
                print('WARNING: BAD NODE SIZE - EVENT LARGER THAN SPECIFIED') 
            if event == 0: ## CONFIG IS EQUAL
                scaler.fit(x)
                upper = x.index.values.tolist() 
                lower = np.roll(upper,-1) 
                lower[len(lower)-1] = len(lower)-1
            x = torch.tensor(scaler.transform(x).tolist(),dtype = torch.float) 
            y = torch.tensor(sca.loc[event,:].drop(columns = 'event_no').values,dtype = torch.float) 
            edge_index = torch.tensor([upper, lower], dtype = torch.long) 
            graphs.append( Data(x = x, edge_index = edge_index,y=  y.unsqueeze(0)))
            print(event)
        s_path = path + '\\' + 'graph_' + file[0:5] + '_' + file[10:15] +'.pki' 
        torch.save(graphs,s_path)
    
    if SRT == False and edge_config == 'Time' and fullgrid == True :
        graphs = list()
        bare_dir = 'J:\\speciale\\data'
        bare_graph = pd.read_csv(bare_dir+'\\bare_graph.csv').loc[:,['dom_x','dom_y','dom_z','dom_charge','dom_time']]
        
        grid = list()
        for k in range(0,len(bare_graph)):
            grid.append(tuple(bare_graph.loc[k,bare_graph.columns[1:4]]))
        grid = pd.Series(grid)
        
        for event in range(0,len(sca)):
            empty_graph = deepcopy(bare_graph)    
            index = seq['event_no'] == sca['event_no'][event]
            x = seq.loc[index,['dom_x',
                               'dom_y',
                               'dom_z',
                               'dom_charge',
                               'dom_time']].sort_values('dom_time').reset_index(drop = True)
            if event == 0:
                scaler.fit(x)
            for node in range(0,len(x)):
                index = grid == tuple(x.loc[node,['dom_x','dom_y','dom_z']])
                if sum(index) == 1:
                    empty_graph.loc[node,['dom_charge','dom_time']] =  x.loc[node,['dom_charge','dom_time']]
                if sum(index) > 1:
                    empty_graph = empty_graph.append(x.loc[node,['dom_x','dom_y','dom_z','dom_charge','dom_time']])
            diff = 6000 - len(empty_graph)
            if diff != 0:
                for j in range(0,diff):
                    empty_graph = empty_graph.append(pd.DataFrame([]*5,columns = ['dom_x','dom_y','dom_z','dom_charge','dom_time']))
            empty_graph = empty_graph.reset_index(drop=True)
            empty_graph = empty_graph.sort_values('dom_time')
            
            upper = empty_graph.index.values.tolist() 
            lower = np.roll(upper,-1) 
            lower[len(lower)-1] = len(lower)-1
            
            x = torch.tensor(scaler.transform(empty_graph).tolist(),dtype = torch.float)
            y = torch.tensor(sca.loc[event,:].drop(columns = 'event_no').values,dtype = torch.float) 
            edge_index = torch.tensor([upper, lower], dtype = torch.long) 
            graphs.append( Data(x = x, edge_index = edge_index,y =  y.unsqueeze(0)))
            print(event)
            
        s_path = path + '\\' + 'graph_' + file[0:5] + '_' + file[10:15] +'full_grid.pki' 
        torch.save(graphs,s_path) 
            
                
                
            
        
    
        
    return print('GRAPHS CREATED: %s || %s '%(s_path,abs(time.time() - now))); 


path = 'J:\\speciale\\results\\40-48-node-split\\test-train'

for file in os.listdir(path): 
    if file.endswith(".csv"): 
        if file[5:9] == '_seq': 
            types_seq = file[0:5] 
            types_sca = 0
        if file[5:9] == '_sca': 
            types_sca = file[0:5]
            types_seq = 0
        for file2 in os.listdir(path):
            types_sca2 = 1
            types_seq2 = 1
            if file.endswith(".csv"): 
                if file2[5:9] == '_seq':
                    types_seq2 = file2[0:5] 
                if file[5:9] == '_sca': 
                    types_sca2 = file[0:5] 
                if types_sca == types_seq2 and file[10:15] == file2[10:15]:
                    sca = pd.read_csv(os.path.join(path, file)) 
                    seq2 = pd.read_csv(os.path.join(path, file2))
                    MakeGraph(seq2, sca, 48, 'J:\\speciale\\results\\graphs\\40-48-split',file=file,fullgrid = True)
                if types_sca2 == types_seq and file[10:15] == file2[10:15]:
                    sca2 = pd.read_csv(os.path.join(path, file2))
                    seq = pd.read_csv(os.path.join(path, file))
                    MakeGraph(seq, sca2, 48, 'J:\\speciale\\results\\graphs\\40-48-split',file=file,fullgrid = True)
