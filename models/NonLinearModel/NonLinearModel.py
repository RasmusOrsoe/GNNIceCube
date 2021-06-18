import torch                                      
import pandas as pd
import numpy as np
import time                                
from torch.nn import MSELoss
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
import matplotlib.pyplot as plt
import os
##############################################################################
def GrabGraphs(path,mode):
    graphs = list()
    for file in os.listdir(path):
        if file[6:11] == mode:
            graphs.append(path + '\\' + file)
    if len(graphs) == 0:
        print('FILES NOT FOUND')
    return graphs

start = time.time()                                                             ## STARTS TIMER FOR LATER USE

class Net(torch.nn.Module):                                                     
    def __init__(self):                                                         # LAG I MODELLEN#                          
        super(Net, self).__init__()
        l1, l2, l3, l4 = 5, 64, 32, 8
                                             #                                  #
        self.relu  = torch.nn.ReLU  (inplace=True)                              # LAG I MODELLEN#                          
        self.pool1 = TopKPooling    (l1   ,ratio = 0.01)
        self.nn1   = torch.nn.RNNCell(l1,l2)                                    # LAG I MODELLEN#                          
        self.pool2 = TopKPooling    (l2   ,ratio = 0.1 )                        #
        self.nn2   = torch.nn.Linear(l2,l3)
        self.pool3 = TopKPooling    (l3   ,ratio = 0.1 )                        #
        self.nn3   = torch.nn.Linear(l3,l4)                                     #
                                                                                # 
    def forward(self, data):                                                    #
        # Get data
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch      #       

        # 1: Pooling layer + Neural layer (NO activation)
        x, edge_index,_,batch,_,_ = self.pool1(x,edge_index,None,batch)         # 
        x = self.nn1(x)                                                         # 

        # 2: Pooling layer + Neural layer (ReLU actication)
        x, edge_index,_,batch,_,_ = self.pool2(x,edge_index,None,batch)         # 
        x = self.relu(self.nn2(x))                                              # 
        
        # 3: Pooling layer + Neural layer (NO activation)
        x, edge_index,_,batch,_,_ = self.pool3(x,edge_index,None,batch)         # 
        x = self.nn3(x)                                          
        return x                                        
                  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # CHOOSING GPU IF AVAILABLE
                                                      # MOUNTS MODEL ON DEVICE

n_reps = 10
batch_size = 32                                                                 #                                                                        
lr = 1e-3                                                                       # PARAMETERS FOR TRAINING AND PREDICTION
n_epochs = 20                                                                   #       ( lr = Learning Rate )

graphs_train = GrabGraphs(path = 'J:\\speciale\\results\\graphs\\40-48-split\\fullgrid'
                    , mode = 'train')

model = Net().to(device)
for p in range(0,n_reps):
    if model is None:
        model = Net().to(device) 
    else:
        del model
        model = Net().to(device)
    
    res = list()
    for k in range(0,len(graphs_train)):
        data_list_train = torch.load(graphs_train[k])                                   # GRABS FIRST Graph-FILE FOR TRAINING
    
        loader = DataLoader(data_list_train, batch_size = batch_size)                         # LOADS Graph-file INTO BATCH FORMAT
        loss_list =list()                                                               # HOLDS LOSS FOR PLOTTING
        #model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)      # OPTIMIZER
        loss_func = MSELoss()                                                           # LOSS FUNCTION
            
        for i in range(0,len(loader)):                                                  # LOOP OVER BATCHES
            data_train = next(iter(loader))                                             # 
            data_train = data_train.to(device)                                          # MOUNTS DATA TO DEVICE
            model.train()                                                               #
            for epoch in range(n_epochs):                                               # LOOP OVER EPOCHS    
                optimizer.zero_grad()                                                   # 
                out = model(data_train)                                                 # ACTUAL TRAINING
                loss = loss_func(out, data_train.y.float())                             #
                loss.backward()                                                         #
                optimizer.step()
                print('BATCH: %s / %s || EPOCH / %s : %s || MSE: %s' %(i, 
                                                                       len(loader),
                                                                       epoch, 
                                                                       n_epochs
                                                                       , loss.data.item()))                                                        #
                loss_list.append(loss.item())
        print('TOTAL TRAINING TIME ON %s GRAPHS: %s' %(len(data_list_train),
                                                       (time.time() - start)/60))
        
        ## PLOT LOSS
        plt.plot(loss_list)
        plt.xlabel('Iterations')
        plt.ylabel('MSE Loss')
        
        
        
        
    graphs_valid = GrabGraphs('J:\\speciale\\results\\graphs\\40-48-split\\fullgrid', 'valid')
    for j in range(0,len(graphs_valid)):    
        data_list_valid = torch.load(graphs_valid[j])    
        loader = DataLoader(data_list_valid, batch_size = batch_size)                   # LOADS THE Graph-file INTO THE BATCH FORMAT 
        start = time.time()                                                             # STARTS TIMER FOR LATER
        acc = 0                                                                         # VARIABLE FOR NMAE CALCULATION
        for i in range(0,len(loader)):                                                  #
            data_pred = next(iter(loader))                                              # BELOW IS SCALING OF NODE FEATURES        
            data = data_pred.to(device)
            model.eval()                                                                #    
            pred = model(data)                                                          # PREDICTION AND CALCULATION OF NMAE-SCORE
            correct = data.y                                                            #
            acc = acc + (abs(pred - data.y)/abs(data.y)).detach().cpu().numpy()         #
            print('PREDICTING: %s /  %s' %(i+1,len(loader)))                                                                        
        res.append(acc.sum(0)/(batch_size*len(loader)))
    pd.DataFrame(res).to_csv('J:\\speciale\\results\\runs\\fullgrid\\%s.csv'%p, index = False)