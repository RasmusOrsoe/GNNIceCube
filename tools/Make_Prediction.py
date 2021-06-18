import os
import torch
from torch_geometric.data import DataLoader
import numpy as np
import torch.multiprocessing
import math
import torch                                      
import pandas as pd
import numpy as np
import time                                
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from torch_scatter import scatter_min
from torch_scatter import scatter_max
torch.autograd.set_detect_anomaly(True)
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
import torch_geometric
from torch_geometric.nn import ChebConv
from playsound import playsound
from torch_geometric.nn import GatedGraphConv
import matplotlib.pyplot as plt
from torch.nn import MultiLabelMarginLoss
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax



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
                               

class Net(torch.nn.Module):                                                     
    def __init__(self):                                                                                   
        super(Net, self).__init__()
        l1, l2, l3, l4, l5,l6,l7 = 5,16*2,32*2,42*2,32*2,16*2,2
        
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.ReLU(),torch.nn.Linear(l2,l3),torch.nn.ReLU()).to(device)
        self.conv_max = EdgeConv(self.nn_conv1,aggr = 'max')
        self.conv_mean = EdgeConv(self.nn_conv1,aggr = 'mean')
        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

                                                        
        self.nn1 = torch.nn.Linear(3*l3,l4)                                               
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
                                          
                                                                                
    def forward(self, data):                                                    
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,edge_index = KNNAmp(4, x, batch)           
        x_max = self.conv_max(x,edge_index)
        x_mean = self.conv_mean(x,edge_index)
        x_add = self.conv_add(x,edge_index)

        x = self.relu(torch.cat((x_max,x_mean,x_add),dim=1))
                
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)

        
        #x = self.sigmoid(x)
        return x

           
  

def worker(graphs_train,q,batch_size):
    for item in range(len(graphs_train)):
        #print(f'Working on {item}')
        data_list_train = torch.load(graphs_train[item])
        loader = DataLoader(data_list_train,batch_size = batch_size,drop_last=True)
        loader_it = iter(loader)
        for k in range(0,len(loader)):
            q.put(next(loader_it))
        #q. join()
        #print(f'Finished {item}')
    torch.multiprocessing.current_process().close()

    #q.close()
def Slave_Watcher(slaves,dead_check,k,mini_batches):
    slave_check = 0
    if dead_check == False:
        for slave in slaves:
            slave_check += slave.is_alive()
        if(slave_check == 0):
            print('All Slaves Have Perished At: %s / %s'%(k,mini_batches))
            dead_check = True
    else:
        slave_check = 0
    return dead_check

def Spawn_Slaves(n_workers, graph_list,q,batch_size):
    slaves = []
    if(n_workers > len(graph_list)):
        n_workers = len(graph_list)
    for j in range(n_workers):
        slaves.append(torch.multiprocessing.Process(target=worker, args=([graph_list[j],q,batch_size])))
    for slave in slaves:
        slave.start()
    print('All task requests sent\n', end='')
    return slaves

def Create_Slaves(n_workers,graphs_valid,manager,batch_size):
    if(n_workers > len(graphs_valid)):
        graph_list = np.array_split(graphs_valid,len(graphs_valid))
    else:
        graph_list = np.array_split(graphs_valid,n_workers)
    q_val = manager.Queue()
    val_slaves = Spawn_Slaves(n_workers,graph_list,q_val,batch_size)
    return q_val,val_slaves   



def Predict(model,prediction_graphs,n_workers,pred_mini_batches,batch_size,o,handle,baseline):
    print('PREDICTING: \n \
          model   : %s \n \
          n_events: %s' %(baseline,pred_mini_batches*batch_size))
    os.makedirs(r'X:\speciale\results\%s\%s'%(handle,baseline),exist_ok = True)
    torch.save(model,r'X:\speciale\results\%s\%s\model.pkl'%(handle,baseline))
    sm = Softmax(dim = 1)
    predictions     = []
    truths          = []
    pred_events     = []
    E               = []
    manager         = torch.multiprocessing.Manager()
    q               = manager.Queue()
    graph_list      = np.array_split(prediction_graphs,n_workers)
    slaves          = Spawn_Slaves(n_workers, graph_list, q,batch_size)
    count           = torch.tensor([0]).to(device)
    dead_check      = False
    model.eval()
    with torch.no_grad():
        for mini_batch in range(0,pred_mini_batches):
            data            = GrabBatch(q)
            prediction      = model(data)
            print(data)
            pred_events.extend(data.event_no.detach().cpu().numpy())
            predictions.extend(sm(prediction).detach().cpu().numpy())
            count +=1
            dead_check = Slave_Watcher(slaves, dead_check, mini_batch, pred_mini_batches)
            if( count.item() == 10 or mini_batch + 1 == pred_mini_batches):
                print('PREDICTED: %s / %s'%((mini_batch + 1)*batch_size,pred_mini_batches*batch_size))
                count = torch.tensor([0]).to(device)
            
        if( dead_check == False):
            print('Prediction Slaves Still Alive. Getting ChainSaw..')
            for slave in slaves:
                slave.terminate()
        
        print('Saving results...')
        truths          = pd.DataFrame(np.array(truths).squeeze())
        predictions     = pd.DataFrame(np.array(predictions).squeeze())
        pred_events     = pd.DataFrame(pred_events)
        E               = pd.DataFrame(E)
        
        result          = pd.concat([pred_events, predictions],axis = 1)
        result.columns  = ['event_no','muon_pred','v']
        result.to_csv(r'X:\speciale\results\%s\%s\results.csv'%(handle,baseline))
        print('Result Saved at: \n \
              reports: NOT FILED \
              archive: %s '%(r'J:\speciale\results\%s\%s'%(handle,baseline)))
        
        
        
        
        
def GrabGraphs(path):
    graphs = list()
    for file in os.listdir(path):
        graphs.append(path + '\\' + file)
    if len(graphs) == 0:
        print('FILES NOT FOUND')
    return graphs

def GrabBatch(q):
    queue_empty = q.empty()
    while(queue_empty):
        queue_empty = q.empty()
    mini_batch =q.get() 
    return mini_batch.to(device)
    
def Calculate_Minibatches(graphs_pred,handle,batch_size):
    path = r'X:\speciale\data\graphs'
    check = os.path.isfile(path + '\\' + handle + r'\minibatch\minibatch_GRU_pred.csv')
    if(check == False):
        print('Mini-batch information for training not found. Generating..')
        os.makedirs(path + '\\' + handle + r'\minibatch',exist_ok = True)
        mini_batches = 0
        for k in range(0,len(graphs_pred)):
            print('%s / %s'%(k,len(graphs_pred)))
            data_list_pred = torch.load(graphs_pred[k])                                                                                                                                                  
            loader = DataLoader(data_list_pred, batch_size = batch_size,drop_last = True)
            mini_batches +=len(loader)
        pd.DataFrame([mini_batches]).to_csv(path + '\\' + handle + r'\minibatch\minibatch_GRU_pred.csv',)
    else:
        print('Mini-batch information for training found at \n %s'%(path + '\\' + handle + r'\minibatch\minibatch_GRU_pred.csv'))
        mini_batches = pd.read_csv(path + '\\' + handle + r'\minibatch\minibatch_GRU_pred.csv').values[0][1]
        

    return mini_batches
        
        
###########################
#       CONFIGURATION     #
###########################
if __name__ == '__main__':
    
    model_path       = r'X:\speciale\results\dev_classification_000\event_only_classification_muon_neutrino\dynedgev3-LeakyReLu-classification'
    trained_model   = torch.load(model_path + '\model.pkl').to(device)
    
    handle          = 'dev_level2_real_data_000\event_only_real_data_10mio'
    base            = 'dynedgev3-muon_neutrino_classification'
    scheduler       = 'LinearSchedule'    
    batch_size      = 1024
    n_workers       = 5                                                          
    n_reps          = 1                                                                                                                                    
    lr              = 7e-4
    max_lr          = 2.5e-3
    end_lr          = 7e-4                                                                 
    n_epochs        = 30 #1990
    patience        = 5
    loss_decimals   = 10
    
    weights                  = pd.read_csv(r'X:\speciale\data\graphs\standard\sliced\event_only_shuffle(0,1)(1,4)_retro\weights.csv')
    graphs_pred              = np.array(GrabGraphs(path = r'X:\speciale\data\graphs\%s\predict'%handle))
    mini_batches_predict     = Calculate_Minibatches(graphs_pred, handle, batch_size)
    max_iter                 = mini_batches_predict*n_epochs
    steps_up                 =  0.1*max_iter
    steps_down               = max_iter  - steps_up  
    o                        = '%s_bs%s_%s_epoch%s_initlr%s_maxlr%s_endlr%s_ES%s_su%s_sd%s'%(base,batch_size,scheduler,n_epochs,'{:.1e}'.format(lr),
                                                                    '{:.1e}'.format(max_lr),'{:.1e}'.format(end_lr),patience,steps_up,steps_down)  
    
    ### CONFIGURATION MESSAGE
    config = 'CONFIGURATION:\n \
          base: %s \n \
          n_reps: %s \n \
          batch_size: %s \n \
          learning rate: %s \n \
          max_lr: %s \n \
          end_lr: %s \n \
          n_epochs: %s \n \
          patience: %s \n \
          scheduler: %s \n \
          steps_up: %s \n \
          steps_down: %s '%(base,n_reps,batch_size,'{:.1e}'.format(lr),'{:.1e}'.format(max_lr),'{:.1e}'.format(end_lr),n_epochs,patience,scheduler,steps_up,steps_down)
    print(config)
    
    ### Prediction

    Predict(trained_model,graphs_pred,n_workers,mini_batches_predict,batch_size,o,handle,base)
    
    playsound(r'J:\speciale\resources\warpcorecollapse.mp3')