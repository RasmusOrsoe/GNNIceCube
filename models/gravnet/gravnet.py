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
#from torch_geometric.nn import GravNetConv2 as GravNetConv

##############################################################################
from typing import Optional, Union
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor

import torch
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing

try:
    from torch_cluster import knn
except ImportError:
    knn = None


class GravNetConv(MessagePassing):
    r"""The GravNet operator from the `"Learning Representations of Irregular
    Particle-detector Geometry with Distance-weighted Graph
    Networks" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is
    dynamically constructed using nearest neighbors.
    The neighbors are constructed in a learnable low-dimensional projection of
    the feature space.
    A second projection of the input feature space is then propagated from the
    neighbors to each vertex using distance weights that are derived by
    applying a Gaussian function to the distances.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        space_dimensions (int): The dimensionality of the space used to
           construct the neighbors; referred to as :math:`S` in the paper.
        propagate_dimensions (int): The number of features to be propagated
           between the vertices; referred to as :math:`F_{\textrm{LR}}` in the
           paper.
        k (int): The number of nearest neighbors.
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 space_dimensions: int, propagate_dimensions: int, k: int,
                 num_workers: int = 1, **kwargs):
        super(GravNetConv, self).__init__(flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`GravNetConv` requires `torch-cluster`.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.num_workers = num_workers

        self.lin_s = Linear(in_channels, space_dimensions)
        self.lin_h = Linear(in_channels, propagate_dimensions)
        self.lin = Linear(in_channels + 2 * propagate_dimensions, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_s.reset_parameters()
        self.lin_h.reset_parameters()
        self.lin.reset_parameters()

    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        """"""

        is_bipartite: bool = True
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
            is_bipartite = False
        assert x[0].dim() == 2, 'Static graphs not supported in `GravNetConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        h_l: Tensor = self.lin_h(x[0])

        s_l: Tensor = self.lin_s(x[0])
        s_r: Tensor = self.lin_s(x[1]) if is_bipartite else s_l

        edge_index = knn(s_l, s_r, self.k, b[0], b[1])

        edge_weight = (s_l[edge_index[1]] - s_r[edge_index[0]]).pow(2).sum(-1)
        edge_weight = torch.exp(-10. * edge_weight)  # 10 gives a better spread

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=(h_l, None),
                             edge_weight=edge_weight,
                             size=(s_l.size(0), s_r.size(0)))

        return self.lin(torch.cat([out, x[1]], dim=-1))

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j * edge_weight.unsqueeze(1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        out_mean = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce='mean')
        out_max = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                          reduce='max')
        return torch.cat([out_mean, out_max], dim=-1)

    def __repr__(self):
        return '{}({}, {}, k={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.k)


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
def NMAELoss(output,target,weight):
    #loss = torch.sum(weight.to(device)*torch.abs((output-target)/target))
    #out[out>4] = 4
    #out[out<1] = 1
    loss = torch.sum(torch.abs((output-target)/target))
    
    return loss
start = time.time()                                                            

class Net(torch.nn.Module):                                                     
    def __init__(self):                                                                                   
        super(Net, self).__init__()
        l1, l2, l3, l4, l5,l6,l7,l8,l9 = 5,32, 64,96,124,64,32,16,1
        self.conv1 = GravNetConv(l1, l2, 3,
                              l1, 5)
        self.conv2 = GravNetConv(l2, l3,3,l2,5)
        self.conv3 = GravNetConv(l4,l5,3,l4,5)
        self.conv4 = GCNConv(l6,l7)
        self.pool1 = TopKPooling    (l3   ,ratio = 0.65)                                                         
        self.nn1 = torch.nn.Linear(l3,l4)                                               
        self.nn2   = torch.nn.Linear(l4*4,l6)
        self.nn3 =  torch.nn.Linear(l7,l8)
        self.nn4 =torch.nn.Linear(l8,l9)
        self.tanh = torch.nn.ReLU()
                                          
                                                                                
    def forward(self, data):                                                    
        x, edge_index, batch = data.x, data.edge_index, data.batch
                   
        x = self.conv1(x,batch)
        x = self.tanh(x)
        x = self.conv2(x,batch)
        #x, edge_index,_,batch,_,_ = self.pool1(x,edge_index,None,batch)  
        x = self.tanh(x)
        x = self.nn1(x)
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        edge_index = torch.tensor([range(0,batch_size),range(0,batch_size)]).to(device)
        
        #a = self.conv3(a,edge_index)
        #b = self.conv3(b,edge_index)
        #c = self.conv3(c,edge_index)
        #d = self.conv3(d,edge_index)
        
        x = torch.cat((a,b,c,d),dim = 1)
        x = self.tanh(x)
        x = self.nn2(x)
        
        x = self.conv4(x,edge_index)
        x = self.tanh(x)
        x = self.nn3(x)
        x = self.tanh(x)
        x = self.nn4(x)
        

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # CHOOSING GPU IF AVAILABLE
                                                      # MOUNTS MODEL ON DEVICE
sca = pd.read_csv(r'J:\speciale\data\raw\standard\sliced\scalar.csv')
sca = sca.loc[:,sca.columns[1:9]]
scalers = list()
for c in sca.columns:
    scaler = MinMaxScaler(feature_range=(1,4)).fit(np.array(sca.loc[:,c]).reshape(-1,1))
    torch.save(scaler,r'J:\speciale\data\minmax(1,10)scaler\scaler_%s.pkl' %c)
    scalers.append(scaler)
n_reps = 1
batch_size =  100                                                             #                                                                        
lr = 1e-5                                                                      # PARAMETERS FOR TRAINING AND PREDICTION
n_epochs = 20                                                                   #       ( lr = Learning Rate )
o = 'grav_net_full-plain+'
graphs_train = GrabGraphs(path = 'J:\\speciale\\data\\graphs\\standard\\sliced\\event_only_even_shuffle_(0,1)\\train')
                                                                                        #, mode = 'train')
scaler = torch.load(r'J:\speciale\data\minmax(1,10)scaler\target\scaler_E_special.pkl')

weights = pd.read_csv(r'J:\speciale\data\graphs\standard\sliced\weights\weights.csv')

model = Net().to(device)
for p in range(0,n_reps):
    if model is None:
        model = Net().to(device) 
    else:
        del model
        model = Net().to(device)
    
    res = list()
    count = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) 
    for epoch in range(n_epochs):
        for k in range(0,len(graphs_train)):
            data_list_train = torch.load(graphs_train[k])                                   # GRABS FIRST Graph-FILE FOR TRAINING
            loss_list =list()                                                               # HOLDS LOSS FOR PLOTTING# OPTIMIZER
            loss_func = NMAELoss                                                           # LOSS FUNCTION
            loader = DataLoader(data_list_train, batch_size = batch_size)                         # LOADS Graph-file INTO BATCH FORMAT
            loader_it = iter(loader)
            for i in range(0,len(loader)):                                                  # LOOP OVER BATCHES
                data_train = next(loader_it)# 
                data_train.y = torch.tensor(scaler.transform(np.array(data_train.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device) #data_train.y[:,0].unsqueeze(1)
                data_train = data_train.to(device)                                          # MOUNTS DATA TO DEVICE
                model.train()                                                               #
                w = 1 #w = torch.tensor(np.array(weights[len(data_train.y)*i:len(data_train.y)*(i+1)]))  
                optimizer.zero_grad()                                                   # 
                out = model(data_train)                                                 # ACTUAL TRAINING
                loss = loss_func(out, data_train.y.float(),w)                             #
                loss.backward()                                                         #
                optimizer.step()
                
                nmae_print = CalculateNMAE(out,data_train.y)
                count =  count + 1
                print('BATCH: %s / %s || EPOCH / %s : %s || MSE: %s || NMAE: %s' %(count, 
                                                                       len(loader)*len(graphs_train)*n_epochs,
                                                                       epoch, 
                                                                       n_epochs
                                                                       , round(loss.data.item(),4), 
                                                                      nmae_print))                                                        #
                loss_list.append(loss.item())
            print('TOTAL TRAINING TIME ON %s GRAPHS: %s' %(len(data_list_train),
                                                           (time.time() - start)/60))   
graphs_valid = GrabGraphs('J:\\speciale\\data\\graphs\\standard\\sliced\\event_only_even_shuffle_(0,1)\\valid')
count = 0
target = list() 
res = list()
model.eval() 
for j in range(0, int(len(graphs_valid))):
    data_train = 0;
    loader = 0;
    data_list_train = 0;
    data_list_valid = torch.load(graphs_valid[j])   
    loader = DataLoader(data_list_valid, batch_size = batch_size)                   # LOADS THE Graph-file INTO THE BATCH FORMAT 
    loader_it = iter(loader)
    start = time.time()                                                             # STARTS TIMER FOR LATER                                                                        # VARIABLE FOR NMAE CALCULATION
    with torch.no_grad():
        for i in range(0, len(loader)):                                                  #
            count = count + 1
            data_pred = next(loader_it)                                              # BELOW IS SCALING OF NODE FEATURES        
            data_pred.y = torch.tensor(scaler.transform(np.array(data_pred.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device)#data_pred.y[:,0].unsqueeze(1)
            data = data_pred.to(device)
                                                                           #    
            pred = model(data)                                                          # PREDICTION AND CALCULATION OF NMAE-SCORE
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
pd.DataFrame([str(model)]).to_csv(r'J:\speciale\results\runs\results\event_only\%s_conf.txt'%o, index = False)
pd.DataFrame(result).to_csv(r'J:\speciale\results\runs\results\event_only\%s_result.csv'%o, index = False)

    