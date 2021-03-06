import torch                                      
import pandas as pd
import numpy as np
import time                                
from torch.nn import MSELoss
#from torch.nn import MAELoss
from torch.nn import CrossEntropyLoss
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
import matplotlib.pyplot as plt
import os
from torch_geometric.nn import ARMAConv
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from torch_scatter import scatter_min
from torch_scatter import scatter_max
torch.autograd.set_detect_anomaly(True)
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch_cluster import knn_graph
from torch_geometric.nn import LEConv
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CyclicLR
import torch.nn.utils.rnn as rnn
##############################################################################
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.inits import reset

## CONST
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
count = torch.tensor([0]).to(device)

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

class lr_watcher:
    def __init__(self, start_lr, max_lr, min_lr, n_rise, n_fall, batch_size, schedule='exp', iterations_completed=1):
        """Calculates the factor the initial learning rate should be multiplied with to get desired learning rate. Options: 'inverse', 'exp'
        Arguments:
            start_lr {float} -- initial learning rate
            max_lr {float} -- maximal learning rate during training
            min_lr {float} -- minimal/end learning rate during training
            n_rise {int} -- steps up from initial learning rate
            n_fall {int} -- steps down from max learning rate
            batch_size {int} -- used batch size
        Keyword Arguments:
            schedule {str} -- Keyword for factor calculation (default: {'exp'})
        """        
        # To ensure no nasty divisions by 0
        self._steps_up = max(n_rise//batch_size, 1)
        self._steps_down = max(n_fall//batch_size, 1)
        self.gamma_up = (max_lr/start_lr)**(1/self._steps_up)
        self.gamma_down = (min_lr/max_lr)**(1/self._steps_down)
        self._start_lr = start_lr
        self._max_lr = max_lr
        self.schedule = schedule
        self.step = iterations_completed
        if schedule == 'inverse':
            # 1/t decay
            frac = min_lr/max_lr
            self.s = self._steps_down*frac/(1-frac)
    def get_factor(self):
        if self.schedule == 'exp':
            if self.step < self._steps_up:
                factor = self.gamma_up**self.step
            else:
                factor = (self._max_lr/self._start_lr) * self.gamma_down**(self.step-self._steps_up)
            self.step += 1
        elif self.schedule == 'inverse':
            if self.step < self._steps_up:
                factor = self.gamma_up**self.step
            else:
                factor = (self._max_lr/self._start_lr) * self.s/(self.s+(self.step-self._steps_up))
            self.step += 1
        else:
            raise ValueError('lr_watcher: Unknown (%s) schedule given!'%(self.schedule))
        return factor
def LinearSchedule(lr,lr_max,lr_min,steps_up,steps_down):
    res = list()
    for step in range(0,steps_up+steps_down):
        slope_up = (lr_max - lr)/steps_up
        slope_down = (lr_min - lr_max)/steps_down
        if step <= steps_up:
            res.append(step*slope_up + lr)
        if step > steps_up:
            res.append(step*slope_down + lr_max -((lr_min - lr_max)/steps_down)*steps_up)
    return(res)
def KNNAmp(k,x,batch):
    pos = x[:,0:3]
    edge_index = knn_graph(x=pos,k=k,batch=batch).to(device)
    nodes = list()
    #for i in batch.unique():
    #    nodes = (batch == i).sum().item()
    #    index = batch == i
    #    x[index,3:5] = x[index,3:5]*nodes
    return x,edge_index
##############################################################################
def GrabGraphs(path):
    graphs = list()
    for file in os.listdir(path):
        graphs.append(path + '\\' + file)
    if len(graphs) == 0:
        print('FILES NOT FOUND')
    return graphs
def CalculateNMAE(pred,target):
    nmae = (abs(pred-target)/target).mean().data.item()
    return nmae
def schedule(epoch):
    global schedule_count
    upper = 7*1e-4
    lower = 2*1e-3
    up = np.arange(upper,lower,0.0005)
    down = up[::-1].sort()
    return()

def NMAELoss(output,target,weight):
    #loss = torch.sum(weight.to(device)*torch.abs((output-target)/target))
    #out[out>4] = 4
    #out[out<1] = 1
    #loss = torch.sum(torch.abs((output-target)/target))
    loss = torch.sum(torch.abs((output-target)))
    
    return loss
start = time.time()                                                            

class Net(torch.nn.Module):                                                     
    def __init__(self):                                                                                   
        super(Net, self).__init__()
        l1, l2, l3, l4, l5,l6,l7 = 5,64,128,94,64,32,1
        self.hidden_dim = 3*l3
        self.n_layers = 2
        n_layers = self.n_layers
        
        self.gru = torch.nn.GRU(4*l5,l6, n_layers, dropout=0)
        
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.ReLU(),torch.nn.Linear(l2,l3),torch.nn.ReLU()).to(device)
        self.conv_max = EdgeConv(self.nn_conv1,aggr = 'max')
        self.conv_mean = EdgeConv(self.nn_conv1,aggr = 'mean')
        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

                                                        
        self.nn1 = torch.nn.Linear(3*l3,l4)                                               
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.ReLU()
                                          
                                                                                
    def forward(self, data,h_1):                                                    
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,edge_index = KNNAmp(4, x, batch)           
        x_max = self.conv_max(x,edge_index)
        x_mean = self.conv_mean(x,edge_index)
        x_add = self.conv_add(x,edge_index)

        x = self.relu(torch.cat((x_max,x_mean,x_add),dim=1))      
        x = self.relu(x)
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        
        
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        x = self.relu(x)
        #x = self.nn3(x)  ## 4*l5
    
        
        seq_length = int(len(x)/batch_size)
        ha = 0
        x = torch.reshape(x,(seq_length,x.shape[1],batch_size)).transpose(1,2)
        x, h = self.gru(x, h_1)
        x = torch.reshape(x.transpose(1,2),(-1,x.shape[2]))
        x = self.relu(x)
        x = self.nn4(x)
        x = self.relu(x)
        return x, h
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, 83200, self.hidden_dim).zero_().to(device)
        return None

           
                                                     
#%%

graphs_train = GrabGraphs(path = r'J:\speciale\data\graphs\dev_numu_train_l5_retro_001\event_only_shuffled(0,1)(1,4)_exp\train')
weights = pd.read_csv(r'J:\speciale\data\graphs\standard\sliced\weights\weights.csv')

 
##CONFIG                                                          
n_reps = 1
batch_size = 1024                                                                                                                                    
lr = 1e-4
max_lr = 9e-3
end_lr = 1e-5                                                                 
n_epochs = 1990
patience = 5
mini_batches = 0
for k in range(0,len(graphs_train)):
    data_list_train = torch.load(graphs_train[k])                                                                                                                                                  
    loader = DataLoader(data_list_train, batch_size = batch_size)
    mini_batches +=len(loader)
max_iter = mini_batches*n_epochs
steps_up =  0.1*max_iter
steps_down = max_iter  - steps_up  
es = EarlyStopping(patience=patience)
base = 'dynedgeGRU_baseline_nlay=1_test'
scheduler = 'BjørnInverse'
o = '%s_bs%s_%s_epoch%s_initlr%s_maxlr%s_endlr%s_ES%s_su%s_sd%s'%(base,batch_size,scheduler,n_epochs,'{:.1e}'.format(lr),
                                                                '{:.1e}'.format(max_lr),'{:.1e}'.format(end_lr),patience,steps_up,steps_down)  
##########
lr_factor = list()
for k in range(0,int(max_iter)):
   lr_factor.append(lr_watcher(lr,max_lr,end_lr,steps_up,steps_down,batch_size,schedule = 'inverse',iterations_completed = k+1).get_factor())


lr_list = torch.tensor(np.array(lr_factor)*lr)

#lr_list =  torch.tensor(LinearSchedule(lr,max_lr,end_lr,steps_up = mini_batches*2, steps_down = mini_batches*2000)).to(device)

##STATICSs
#%%
loss_list = list()
epoch_list = list()                                                                                  
count = torch.tensor([0]).to(device)
model = Net().to(device)
### CONFIGURATION MESSAGE
config = 'CONFIGURATION:\n \
      n_reps: %s \n \
      batch_size: %s \n \
      learning rate: %s \n \
      max_lr: %s \n \
      end_lr: %s \n \
      n_epochs: %s \n \
      patience: %s \n \
      scheduler: %s \n \
      steps_up: %s \n \
      steps_down: %s '%(n_reps,batch_size,'{:.1e}'.format(lr),'{:.1e}'.format(max_lr),'{:.1e}'.format(end_lr),n_epochs,patience,scheduler,steps_up,steps_down)
print(config)
### TRAINING
for p in range(0,n_reps):
    if model is None:
        model = Net().to(device) 
    else:
        del model
        model = Net().to(device)
    
    res = list()
    count = torch.tensor(0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = CyclicLR(optimizer,base_lr =7*1e-5,max_lr = 2*1e-4,step_size_up = 25000,cycle_momentum = False)
    print('TRAINING BEGUN. FIRST EPOCH STARTING..')
    h_in = model.init_hidden(batch_size)
    for epoch in range(n_epochs):
        loss_acc = torch.tensor([0],dtype = float).to(device)
        for k in range(0,len(graphs_train)):
            data_list_train = torch.load(graphs_train[k])                                                                                         
            loss_func = NMAELoss                                                          
            loader = DataLoader(data_list_train, batch_size = batch_size,drop_last=True)                         
            loader_it = iter(loader)
            for i in range(0,len(loader)):                                                 
                data_train = next(loader_it).to(device) 
                data_train.y = data_train.y[:,0].unsqueeze(1)#torch.tensor(scaler.inverse_transform(np.array(data_train.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device) #data_train.y[:,0].unsqueeze(1)
                #data_train = data_train.to(device)                                         
                model.train()                                                               
                w = 1 #w = torch.tensor(np.array(weights[len(data_train.y)*i:len(data_train.y)*(i+1)]))  
                optimizer.zero_grad()
                out,h_0 = model(data_train,h_in)
                loss = loss_func(out, data_train.y.float(),w)                             
                loss.backward()                                                         
                optimizer.step()
                h_in = h_0.detach().clone()
                #loss_list.append(loss.item())
                #epoch_list.append(optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = lr_list[count].item()
                count +=1
                loss_acc +=loss
                #print(i)
                #scheduler.step()
               
        if epoch == 0:
            deltatime = (time.time() - start)/60
        print('EPOCH: %s / %s || %s / %s min || LR: %s || Loss: %s || iter: %s/%s' %(epoch,n_epochs,(time.time() - start)/60,n_epochs*deltatime,optimizer.param_groups[0]['lr'],round(loss_acc.item()/(mini_batches*batch_size),3),count.item(),max_iter))
        if es.step(torch.tensor([round(loss_acc.item()/(mini_batches*batch_size),3)])):
            print('EARLY STOPPING: %s'%epoch)
            break
        
#=============================================================================
graphs_valid = GrabGraphs(r'J:\speciale\data\graphs\dev_numu_train_l5_retro_001\event_only_shuffled(0,1)(1,4)_exp\valid')
count = 0
target = list() 
res = list()
model.eval() 
for j in range(0, int(len(graphs_valid))):
    data_train = 0;
    loader = 0;
    data_list_train = 0;
    data_list_valid = torch.load(graphs_valid[j])   
    loader = DataLoader(data_list_valid, batch_size = batch_size,drop_last = True)                   # LOADS THE Graph-file INTO THE BATCH FORMAT 
    loader_it = iter(loader)
    #start = time.time()                                                             # STARTS TIMER FOR LATER                                                                        # VARIABLE FOR NMAE CALCULATION
    with torch.no_grad():
        for i in range(0, len(loader)):                                                  #
            count = count + 1
            data_pred = next(loader_it)                                              # BELOW IS SCALING OF NODE FEATURES        
            data_pred.y = data_pred.y[:,0].unsqueeze(1) #torch.tensor(scaler.inverse_transform(np.array(data_pred.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device)#data_pred.y[:,0].unsqueeze(1)
            data = data_pred.to(device)
                                                                            #    
            pred,h_0 = model(data,h_in)                                                          # PREDICTION AND CALCULATION OF NMAE-SCORE
            h_in = h_0.detach().clone()
            print('PREDICTING: %s /  %s' %(count,len(loader)*len(graphs_valid)))                                                                        
            res.extend(pred.detach().cpu().numpy())
            target.extend(data.y.detach().cpu().numpy())
    

pred = pd.DataFrame(res)
target = pd.DataFrame(target)
result = pd.concat([abs((pred-target)/target),pred, target],axis = 1)
#result.columns = ['nmae E','nmae T','nmae pos_x','nmae pos_y','nmae pos_z',
#                  'nmae dir_x','nmae dir_y','nmae dir_z',
#                  'E_pred','T_pred','pos_x_pred','pos_y_pred','pos_z_pred',
#                  'dir_x_pred','dir_y_pred','dir_z_pred'
#                  ,'E','T','pos_x','pos_y','pos_z','dir_x','dir_y','dir_z']
result.columns = ['nmae E', 'E_pred','E']
pd.DataFrame([config,str(model)]).to_csv(r'J:\speciale\results\runs\results\event_only\%s_conf.txt'%o, index = False)
pd.DataFrame(result).to_csv(r'J:\speciale\results\runs\results\event_only\%s_result.csv'%o, index = False)
#=============================================================================

print('Total Time Elapsed: %s min'%((time.time() - start)/60))
pd.DataFrame([time.time() - start]).to_csv(r'J:\speciale\results\runs\results\event_only\timed\%s_timed.csv'%o)
pd.DataFrame(loss_list).to_csv(r'J:\speciale\results\runs\results\event_only\%s_loss.csv'%o)
pd.DataFrame(epoch_list).to_csv(r'J:\speciale\results\runs\results\event_only\%s_lr.csv'%o)