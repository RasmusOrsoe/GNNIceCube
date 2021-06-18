import os
import torch
import numpy as np
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

import torch.nn.functional as F

import torch
import torch_geometric

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()


device = torch.device('cpu')

def KNNAmp(k,x,batch):
    pos = x[:,0:3]
    edge_index = knn_graph(x=pos,k=k,batch=batch).to(device)
    nodes = list()
    #for i in batch.unique():
    #    nodes = (batch == i).sum().item()
    #    index = batch == i
    #    x[index,3:5] = x[index,3:5]*nodes
    return x,edge_index


class dynedge_energy(torch.nn.Module):                                                     
    def __init__(self,k):                                                                                   
        super(dynedge_energy, self).__init__()
        c = 3 
        l1, l2, l3, l4, l5,l6,l7 = 8,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,1
        self.k = k
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)
        #self.conv_max = EdgeConv(self.nn_conv1,aggr = 'max')
        #self.conv_mean = EdgeConv(self.nn_conv1,aggr = 'mean')
        #self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

        self.nn_conv2 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add2 = EdgeConv(self.nn_conv2,aggr = 'add')

        self.nn_conv3 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add3 = EdgeConv(self.nn_conv3,aggr = 'add')

        self.nn_conv4 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add4 = EdgeConv(self.nn_conv4,aggr = 'add')

        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                               
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
                                          
                                                                                
    def forward(self, data):   
        k = self.k                                                 
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,edge_index = KNNAmp(k, x, batch)

        a = self.conv_add(x,edge_index)
        
        _,edge_index = KNNAmp(k, a, batch)
        b = self.conv_add2(a,edge_index)

        _,edge_index = KNNAmp(k, b, batch)
        c = self.conv_add3(b,edge_index)

        _,edge_index = KNNAmp(k, c, batch)
        d = self.conv_add4(c,edge_index)

        x2 = torch.cat((x,a,b,c,d),dim = 1) 
        
       


        x = self.nn1(x2)
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

        

        return x

class dynedge_angle(torch.nn.Module):                                                     
    def __init__(self, k):                                                                                   
        super(dynedge_angle, self).__init__()
        c = 3
        l1, l2, l3, l4, l5,l6,l7 = 8,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,3
        self.k = k
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)
        #self.conv_max = EdgeConv(self.nn_conv1,aggr = 'max')
        #self.conv_mean = EdgeConv(self.nn_conv1,aggr = 'mean')
        #self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

        self.nn_conv2 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add2 = EdgeConv(self.nn_conv2,aggr = 'add')

        self.nn_conv3 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add3 = EdgeConv(self.nn_conv3,aggr = 'add')

        self.nn_conv4 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add4 = EdgeConv(self.nn_conv4,aggr = 'add')

        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                               
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
                                          
                                                                                
    def forward(self, data):
        k = self.k                                                    
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,edge_index = KNNAmp(k, x, batch)

        a = self.conv_add(x,edge_index)
        
        _,edge_index = KNNAmp(k, a, batch)
        b = self.conv_add2(a,edge_index)

        _,edge_index = KNNAmp(k, b, batch)
        c = self.conv_add3(b,edge_index)

        _,edge_index = KNNAmp(k, c, batch)
        d = self.conv_add4(c,edge_index)

        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
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
        
        x[:,0] = self.tanh(x[:,0])
        x[:,1] = self.tanh(x[:,1])

        

        return x



class dynedge_classification(torch.nn.Module):                                                     
    def __init__(self, k):                                                                                   
        super(dynedge_classification, self).__init__()
        c = 3
        l1, l2, l3, l4, l5,l6,l7 = 8,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,3
        self.k = k
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)
        #self.conv_max = EdgeConv(self.nn_conv1,aggr = 'max')
        #self.conv_mean = EdgeConv(self.nn_conv1,aggr = 'mean')
        #self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

        self.nn_conv2 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add2 = EdgeConv(self.nn_conv2,aggr = 'add')

        self.nn_conv3 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add3 = EdgeConv(self.nn_conv3,aggr = 'add')

        self.nn_conv4 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add4 = EdgeConv(self.nn_conv4,aggr = 'add')

        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                               
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
                                          
                                                                                
    def forward(self, data):
        k = self.k                                                    
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,edge_index = KNNAmp(k, x, batch)
#        x =  x[:,0:5]
#        print(x.shape)
        a = self.conv_add(x,edge_index)
        
        _,edge_index = KNNAmp(k, a, batch)
        b = self.conv_add2(a,edge_index)

        _,edge_index = KNNAmp(k, b, batch)
        c = self.conv_add2(b,edge_index)

        _,edge_index = KNNAmp(k, c, batch)
        d = self.conv_add2(c,edge_index)

        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
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
        
      
        

        return x



energy = dynedge_energy(k = 8)
angle = dynedge_angle(k = 8)
classification = dynedge_classification(k = 8)

n_energy = sum(p.numel() for p in energy.parameters() if p.requires_grad)
n_angle = sum(p.numel() for p in angle.parameters() if p.requires_grad)
n_classification = sum(p.numel() for p in classification.parameters() if p.requires_grad)

print('energy: %s'%n_energy)
print('angle: %s'%n_angle)
print('classification: %s'%n_classification)