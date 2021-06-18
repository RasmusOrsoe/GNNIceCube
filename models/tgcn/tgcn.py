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
from torch_geometric.nn import ChebConv
from playsound import playsound

class GConvGRUv2(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, K: int,
                 normalization: str="sym", bias: bool=True):
        super(GConvGRUv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()


    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_z = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)


    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_r = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)


    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_h = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)


    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()


    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(device)
        return H


    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = self.conv_x_z(X, edge_index, edge_weight)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight)
        Z = torch.sigmoid(Z)
        return Z


    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = self.conv_x_r(X, edge_index, edge_weight)
        R = R + self.conv_h_r(H, edge_index, edge_weight)
        R = torch.sigmoid(R) 
        return R


    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight)
        H_tilde = H_tilde + self.conv_h_h(H*R, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde


    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z*H + (1-Z)*H_tilde
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
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
    #loss = torch.sum(weight.to(device)*torch.abs((output-target)))
    #out[out>4] = 4
    #out[out<1] = 1
    #loss = torch.sum(torch.abs((output-target)/target))
    loss = torch.sum(torch.abs((output-target)))
    
    return loss
start = time.time()                                                            

class Net(torch.nn.Module):                                                     
    def __init__(self):                                                                                   
        super(Net, self).__init__()
        l1 = 5
        l2 = 64
        l3 = 32
        l4 = 16
        l5 = 1
        kernel = 2
        self.recurrent_1 = GConvGRUv2(l1, l2, kernel)
        self.recurrent_2 = GConvGRUv2(l2, l3, kernel)
        self.recurrent_3 = GConvGRUv2(l3, l4, kernel)
        self.linear = torch.nn.Linear(2*l4, l4)
        self.linear2 = torch.nn.Linear(l4,l5)
                                          
                                                                                

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = torch.FloatTensor(np.random.uniform(1, 1, (edge_index.shape[1]))).to(device)
        x = self.recurrent_1(x, edge_index)
        #x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index)
        #x = F.dropout(x, training=self.training)
        x = self.recurrent_3(x, edge_index)
        #x = F.dropout(x, training=self.training)
        a,_ = scatter_max(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        x = torch.cat((a,c),dim = 1)
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.relu(x)

           
  

def worker(graphs_train,q):
    for item in range(len(graphs_train)):
        #print(f'Working on {item}')
        data_list_train = torch.load(graphs_train[item])
        loader = DataLoader(data_list_train,batch_size = 1024,drop_last=True)
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

def Spawn_Slaves(n_workers, graph_list,q):
    slaves = []
    if(n_workers > len(graph_list)):
        n_workers = len(graph_list)
    for j in range(n_workers):
        slaves.append(torch.multiprocessing.Process(target=worker, args=([graph_list[j],q])))
    for slave in slaves:
        slave.start()
    print('All task requests sent\n', end='')
    return slaves

def Create_Slaves(n_workers,graphs_valid,manager):
    if(n_workers > len(graphs_valid)):
        graph_list = np.array_split(graphs_valid,len(graphs_valid))
    else:
        graph_list = np.array_split(graphs_valid,n_workers)
    q_val = manager.Queue()
    val_slaves = Spawn_Slaves(n_workers,graph_list,q_val)
    return q_val,val_slaves   
        
def Train(model_to_train,graphs_train,device,lr,lr_list,n_workers,batch_size,loss_decimals,graphs_valid,mini_batches_valid):                                                                                
    count = torch.tensor([0]).to(device)
    model = model_to_train.to(device)
    loss_func = NMAELoss
    loss_pr_epoch = []
    for p in range(0,n_reps):
        if model is None:
            model = model_to_train.to(device)
        else:
            del model
            model =  model_to_train.to(device)
        count = torch.tensor(0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print('TRAINING BEGUN. FIRST EPOCH STARTING..')
        
        for epoch in range(0,n_epochs):
            dead_check =  False
            transition_val = False
            loss_acc = torch.tensor([0],dtype = float).to(device)
            if epoch == 0:
                manager = torch.multiprocessing.Manager()
                q = manager.Queue()
                graph_list = np.array_split(graphs_train,n_workers)
                slaves = Spawn_Slaves(n_workers, graph_list, q)
            for k in range(0,mini_batches):                                               
                data_train                      = GrabBatch(q)
                data_train.y                    = data_train.y[:,0].unsqueeze(1)#torch.tensor(scaler.inverse_transform(np.array(data_train.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device) #data_train.y[:,0].unsqueeze(1)                                         
                model.train()                                                               
                w                               = 1 
                optimizer.zero_grad()                                                   
                out                         = model(data_train)
                if(torch.sum(torch.isnan(out)) != 0):
                    print('NAN ENCOUNTERED AT : %s / %s'%(k,mini_batches))
                                                                
                loss                            = loss_func(out, data_train.y.float(),w)                             
                loss.backward()                                                         
                optimizer.step()
                
                optimizer.param_groups[0]['lr'] = lr_list[count].item()
                count                           +=1
                loss_acc                        +=loss
                dead_check = Slave_Watcher(slaves, dead_check, k, mini_batches)
                if(dead_check  == True and transition_val == False):
                    q_val,val_slaves = Create_Slaves(n_workers, graphs_valid, manager)
                    transition_val = True
                #### TENSORBOARD
                writer.add_scalar('train_loss',loss,count)
                writer.add_scalar('lr',lr_list[count-1].item(), count)
                writer.flush()
                ####
            print('entering validation')
            val_loss,q,slaves = Validate(model,graphs_valid,graphs_train,batch_size,o,
                                         q_val,mini_batches_valid,val_slaves,n_workers,mini_batches,manager,epoch,n_epochs,save = False)
            if epoch == 0:
                deltatime                        = (time.time() - start)/60
            #### TENSORBOARD
            writer.add_scalar('val_loss_acc', val_loss.item(),epoch)
            writer.add_scalar('train_loss_acc',loss_acc.item(),epoch)
            writer.flush()
            ####
            print('EPOCH: %s / %s || %s / %s min || LR: %s || Loss: %s || Val Loss: %s' %(epoch,n_epochs,(time.time() - start)/60,n_epochs*deltatime,optimizer.param_groups[0]['lr'],round(loss_acc.item()/(mini_batches*batch_size),loss_decimals),val_loss.item()))
            if es.step(val_loss):
                print('EARLY STOPPING: %s'%epoch)
                for slave in slaves:
                    slave.terminate()
                del q
                break
            loss_pr_epoch.append(loss_acc.item()/(mini_batches*batch_size))
        #slave.join()
        #q.join()
        print('All work completed')
    return model,loss_pr_epoch
def Validate(trained_model,graphs_valid,graphs_train,batch_size,
             o,q,mini_batches_valid,val_slaves,n_workers,mini_batches,manager,train_epoch,n_epochs,save):
    print('in validation')
    model = trained_model
    target = list() 
    res = list()
    val_events = list()
    model.eval() 
    count = 0
    dead_check = False
    transition_val = False
    stop_now = False
    print('Validating..')                                                                                                                                     
    with torch.no_grad():
        for i in range(0, mini_batches_valid):                                                  
            count = count + 1
            data_pred = GrabBatch(q)                                                      
            data_pred.y = data_pred.y[:,0].unsqueeze(1) #torch.tensor(scaler.inverse_transform(np.array(data_pred.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device)#data_pred.y[:,0].unsqueeze(1)
            data = data_pred.to(device) 
            pred = model(data)                                                          # PREDICTION AND CALCULATION OF NMAE-SCORE                                                        
            #print('VALIDATING:%s /  %s' %(count,len(loader)*len(graphs_valid)))                                                                        
            res.extend(pred.detach().cpu().numpy())
            target.extend(data.y.detach().cpu().numpy())
            val_events.extend(data.event_no.detach().cpu().numpy())
            dead_check = Slave_Watcher(val_slaves, dead_check, i, mini_batches)
            if(dead_check  == True and transition_val == False and train_epoch < n_epochs - 1):
                q_train,train_slaves = Create_Slaves(n_workers, graphs_train, manager)
                transition_val = True

            
        if( dead_check == False):
            print('Validation Slaves Still Alive. Getting ChainSaw..')
            for val_slave in val_slaves:
                val_slave.terminate()
            q_train,train_slaves = Create_Slaves(n_workers, graphs_train, manager)
        pred = pd.DataFrame(res)
        target = pd.DataFrame(target)
        result = pd.concat([abs((pred-target)/target),pred, target,pd.DataFrame(val_events)],axis = 1)
        #result.columns = ['nmae E','nmae T','nmae pos_x','nmae pos_y','nmae pos_z',
        #                  'nmae dir_x','nmae dir_y','nmae dir_z',
        #                  'E_pred','T_pred','pos_x_pred','pos_y_pred','pos_z_pred',
        #                  'dir_x_pred','dir_y_pred','dir_z_pred'
        #                  ,'E','T','pos_x','pos_y','pos_z','dir_x','dir_y','dir_z']
        result.columns = ['nmae E', 'E_pred','E','event_no']
        if save == True:
            pd.DataFrame([config,str(model)]).to_csv(r'J:\speciale\results\runs\results\event_only\%s_conf.txt'%o, index = False)
            pd.DataFrame(result).to_csv(r'J:\speciale\results\runs\results\event_only\%s_result.csv'%o, index = False)
         
            print('Total Time Elapsed: %s min'%((time.time() - start)/60))
            pd.DataFrame([time.time() - start]).to_csv(r'J:\speciale\results\runs\results\event_only\timed\%s_timed.csv'%o)
            return
        #pd.DataFrame(loss_list).to_csv(r'J:\speciale\results\runs\results\event_only\%s_loss.csv'%o)
        #pd.DataFrame(epoch_list).to_csv(r'J:\speciale\results\runs\results\event_only\%s_lr.csv'%o)
        else:
           val_loss = torch.sum(abs(torch.tensor(result['E_pred'] - result['E'])))
           return val_loss,q_train,train_slaves
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

def Get_lr_list(lr,max_lr,end_lr,steps_up,steps_down,batch_size,max_iter,device):
    lr_factor = list()
    for k in range(0,int(max_iter)):
       lr_factor.append(lr_watcher(lr,max_lr,end_lr,steps_up,steps_down,batch_size,schedule = 'inverse',iterations_completed = k+1).get_factor())
    
    lr_list = torch.tensor(np.array(lr_factor)*lr)
    return lr_list.to(device)
    
def Calculate_Minibatches(graphs_train,graphs_valid,handle,batch_size):
    path = r'J:\speciale\data\graphs'
    check = os.path.isfile(path + '\\' + handle + r'\minibatch\minibatch_GRU.csv')
    if(check == False):
        print('Mini-batch information for training not found. Generating..')
        os.makedirs(path + '\\' + handle + r'\minibatch',exist_ok = True)
        mini_batches = 0
        for k in range(0,len(graphs_train)):
            print('%s / %s'%(k,len(graphs_train)))
            data_list_train = torch.load(graphs_train[k])                                                                                                                                                  
            loader = DataLoader(data_list_train, batch_size = batch_size,drop_last = True)
            mini_batches +=len(loader)
        pd.DataFrame([mini_batches]).to_csv(path + '\\' + handle + r'\minibatch\minibatch_GRU.csv',)
    else:
        print('Mini-batch information for training found at \n %s'%(path + '\\' + handle + r'\minibatch\minibatch_GRU.csv'))
        mini_batches = pd.read_csv(path + '\\' + handle + r'\minibatch\minibatch_GRU.csv').values[0][1]
        
    check = os.path.isfile(path + '\\' + handle + r'\minibatch\minibatch_valid_GRU.csv')
    if(check == False):
        print('Mini-batch information for validation not found. Generating..')
        os.makedirs(path + '\\' + handle + r'\minibatch',exist_ok = True)
        mini_batches_valid = 0
        for k in range(0,len(graphs_valid)):
            print('%s / %s'%(k,len(graphs_valid)))
            data_list_valid = torch.load(graphs_valid[k])                                                                                                                                                  
            loader = DataLoader(data_list_valid, batch_size = batch_size,drop_last = True)
            mini_batches_valid +=len(loader)
        pd.DataFrame([mini_batches_valid]).to_csv(path + '\\' + handle + r'\minibatch\minibatch_valid_GRU.csv',)
    else:
        print('Mini-batch information for validation found at \n %s'%(path + '\\' + handle + r'\minibatch\minibatch_valid_GRU.csv'))
        mini_batches_valid = pd.read_csv(path + '\\' + handle + r'\minibatch\minibatch_valid_GRU.csv').values[0][1]
    return mini_batches,mini_batches_valid
        
        
###########################
#       CONFIGURATION     #
###########################
if __name__ == '__main__':
    
    handle = 'dev_numu_train_l5_retro_001\event_only_shuffled(0,1)(1,4)_exp_padded'
    base = 'tgcn-test2_extrann_extragru_kernel15'
    scheduler = 'LinearSchedule'
    
    
    batch_size      = 1024
    n_workers       = 5                                                          
    n_reps          = 1                                                                                                                                    
    lr              = 1e-4
    max_lr          = 9e-3
    end_lr          = 1e-5                                                                 
    n_epochs        = 30 #1990
    patience        = 5
    loss_decimals   = 10
    
    weights                  = pd.read_csv(r'J:\speciale\data\graphs\standard\sliced\event_only_shuffle(0,1)(1,4)_retro\weights.csv')
    graphs_train             = np.array(GrabGraphs(path = r'J:\speciale\data\graphs\%s\train'%handle))
    graphs_valid             = GrabGraphs(r'J:\speciale\data\graphs\%s\valid'%handle)
    mini_batches,mini_batches_valid             = Calculate_Minibatches(graphs_train,graphs_valid, handle, batch_size)
    max_iter                 = mini_batches*n_epochs
    steps_up                 =  0.1*max_iter
    steps_down               = max_iter  - steps_up  
    #lr_list                  = Get_lr_list(lr,max_lr,end_lr,steps_up,steps_down,batch_size,max_iter,device)
    lr_list                 =  torch.tensor(LinearSchedule(lr,max_lr,end_lr,steps_up = int(mini_batches/2), steps_down = mini_batches*n_epochs - int(mini_batches/2))).to(device)
    es                       = EarlyStopping(patience=patience)
    o                        = '%s_bs%s_%s_epoch%s_initlr%s_maxlr%s_endlr%s_ES%s_su%s_sd%s'%(base,batch_size,scheduler,n_epochs,'{:.1e}'.format(lr),
                                                                    '{:.1e}'.format(max_lr),'{:.1e}'.format(end_lr),patience,steps_up,steps_down)  
    writer = SummaryWriter(log_dir= "logs/dynedge/tmp/%s"%o)
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
    
    ### TRAINING
    
    trained_model,loss_history = Train(Net(),graphs_train,device,lr,lr_list,n_workers,batch_size,loss_decimals,graphs_valid,mini_batches_valid)
    
    pd.DataFrame(loss_history).to_csv(r'J:\speciale\results\runs\results\event_only\%s_loss.csv'%o)
    ### VALIDATION
    
    #Validate(trained_model,graphs_valid,batch_size,o,save = True)
    
    writer.close()
    playsound(r'J:\speciale\resources\warpcorecollapse.mp3')