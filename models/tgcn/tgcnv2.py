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


class BasicBlock(torch.nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
        kernel_size = 5):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = torch_geometric.nn.BatchNorm
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = torch_geometric.nn.GraphConv(inplanes, planes)
        self.bn1 = norm_layer(planes,eps=1e-02)
        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.conv2 = torch_geometric.nn.GraphConv(planes, planes)
        self.bn2 = norm_layer(planes,eps=1e-02)
        self.downsample = downsample
        self.stride = stride
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x,edge_index): 
        identity = x

        out = self.conv1(x,edge_index)
        #out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out,edge_index)
        #out = self.bn2(out)
        out = self.dropout(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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

    def step(self, metrics,model):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_params = model.state_dict()
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
    def GetBestParams(self):
        return self.best_params

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
        l1 = 11
        l2 = 64
        l3 = 32
        l4 = 24
        l5 = 16
        l6 = 8
        l7 = 1
        numlayers = 5
        self.resblock1 = BasicBlock(l2,l2)
        self.resblock2 = BasicBlock(l3,l3)
        self.recurrent_1 = GatedGraphConv(l2,numlayers)
        #self.recurrent_2 = GConvGRUv2(l2, l3, kernel)
        #self.recurrent_3 = GConvGRUv2(l3, l4, kernel)
        self.linear_init = torch.nn.Linear(l1, l2)
        self.linear = torch.nn.Linear(l2, l3)
        self.linear1 = torch.nn.Linear(l3, l4)
        self.linear2 = torch.nn.Linear(2*l4,l5)
        self.linear3 = torch.nn.Linear(l5,l6)
        self.linear4 = torch.nn.Linear(l6,l7)
        self.pool1 = torch_geometric.nn.TopKPooling(l2,ratio = 0.5)                                  
        self.pool2 = torch_geometric.nn.TopKPooling(l3,ratio = 0.5)
        self.relu = torch.nn.LeakyReLU()
        self.bn1 =torch_geometric.nn.BatchNorm(l3,eps=1e-02)
        self.bn2 =torch_geometric.nn.BatchNorm(l4,eps=1e-02)
        self.dropout = torch.nn.Dropout(0.2)                                                      
                          

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = torch.FloatTensor(np.random.uniform(1, 1, (edge_index.shape[1]))).to(device)
        x = self.linear_init(x)
        x = self.resblock1(x,edge_index)
        x = self.recurrent_1(x, edge_index)
        x = self.relu(x)
       # x, edge_index, _, batch, _, _ = self.pool1(x,edge_index = edge_index,batch = batch)
        x = self.linear(x)
        #x = self.bn1(x)
        x = self.relu(x)
        #x, edge_index, _, batch, _, _ = self.pool2(x,edge_index = edge_index,batch = batch)
        x = self.linear1(x)
        #x = self.bn2(x)
        x = self.relu(x)
        a,_ = scatter_max(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        x = torch.cat((a,c),dim = 1)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return self.relu(x)

           
  

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

def LRFinder(n_reps,model,limits,loss_function,graphs_train,batch_size,mini_batches,n_workers):
    model_LR = model.to(device)
    loss_func = loss_function
    lr_list = np.linspace(limits[0],limits[1],mini_batches)
    result = []
    for i in range(0,n_reps):
        loss_data = []
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[0],eps = 1e-3)
        print('Learning Rate Finder Initialized. EPOCH STARTING..')
        dead_check =  False
        manager = torch.multiprocessing.Manager()
        q = manager.Queue()
        graph_list = np.array_split(graphs_train,n_workers)
        slaves = Spawn_Slaves(n_workers, graph_list, q)
        for k in range(0,mini_batches):                                               
            data_train                      = GrabBatch(q)
            data_train.y                    = data_train.y[:,0].unsqueeze(1)#torch.tensor(scaler.inverse_transform(np.array(data_train.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device) #data_train.y[:,0].unsqueeze(1)                                         
            model_LR.train()                                                               
            w                               = 1 
            optimizer.zero_grad()                                                   
            out                         = model(data_train)
            if(torch.sum(torch.isnan(out)) != 0):
                print('NAN ENCOUNTERED AT : %s / %s'%(k,mini_batches))
                                                            
            loss                            = loss_func(out, data_train.y.float(),w)                             
            loss.backward()                                                         
            optimizer.step()
            optimizer.param_groups[0]['lr'] = lr_list[k].item()
            dead_check = Slave_Watcher(slaves, dead_check, k, mini_batches)
            loss_data.append(loss.item())
        print('Learning Rate Finder Done, killing slaves..')
        for slave in slaves:
            slave.terminate()
        print('Slaves Are Gone.')
        result.append(loss_data)
          
    return result,lr_list
def LoadBestParams(model,params):
    print('Fetching best model Parameters..')
    model.load_state_dict(params)
    print('Parameters loaded into model. Ready to predict!')
    return model        
def Train(model_to_train,graphs_train,device,lr,lr_list,
          n_workers,batch_size,loss_decimals,graphs_valid,
          mini_batches_valid):                                                                                
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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,eps = 1e-3)
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
            if( dead_check == False):
                print('Training Slaves Still Alive. Getting ChainSaw..')
                for slave in slaves:
                    slave.terminate()
                q_val,val_slaves =  Create_Slaves(n_workers, graphs_valid, manager)
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
            if es.step(val_loss,model):
                print('EARLY STOPPING: %s'%epoch)
                for slave in slaves:
                    slave.terminate()
                del q
                break
            loss_pr_epoch.append(loss_acc.item()/(mini_batches*batch_size))
        dead_check = Slave_Watcher(slaves, dead_check, k, mini_batches)
        if(dead_check == False):
            print('Training Done, killing slaves..')
            for slave in slaves:
                slave.terminate()
            print('Slaves Are Gone.')
        model = LoadBestParams(model,es.GetBestParams())
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
    acc_loss = torch.tensor([0],dtype = float).to(device)
    q_train = None                                                                                                                                    
    with torch.no_grad():
        for i in range(0, mini_batches_valid):                                                  
            count = count + 1
            data_pred = GrabBatch(q)                                                      
            data_pred.y = data_pred.y[:,0].unsqueeze(1) #torch.tensor(scaler.inverse_transform(np.array(data_pred.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device)#data_pred.y[:,0].unsqueeze(1)

            pred = model(data_pred) 
            loss = NMAELoss(pred,data_pred.y,1)
            acc_loss += loss                                                          # PREDICTION AND CALCULATION OF NMAE-SCORE                                                        
            dead_check = Slave_Watcher(val_slaves, dead_check, i, mini_batches)
            if(dead_check  == True and transition_val == False and train_epoch < n_epochs - 1):
                q_train,train_slaves = Create_Slaves(n_workers, graphs_train, manager)
                transition_val = True

            
        if( dead_check == False or q_train == None):
            print('Validation Slaves Still Alive. Getting ChainSaw..')
            for val_slave in val_slaves:
                val_slave.terminate()
            q_train,train_slaves = Create_Slaves(n_workers, graphs_train, manager)
     
        val_loss = acc_loss
        return val_loss,q_train,train_slaves
def Predict(model,prediction_graphs,n_workers,pred_mini_batches,batch_size,o,handle,baseline):
    path     = r'X:\speciale\results\runs\results\event_only'
    print('PREDICTING: \n \
          model   : %s \n \
          n_events: %s' %(baseline,pred_mini_batches*batch_size))
    os.makedirs(r'X:\speciale\results\%s\%s'%(handle,baseline),exist_ok = True)
    torch.save(model.state_dict,r'X:\speciale\results\%s\%s\state_dict.pkl'%(handle,baseline))
    #torch.save(model.parameters(),r'J:\speciale\results\%s\%s\parameters.pkl'%(handle,baseline))
    predictions     = []
    truths          = []
    pred_events     = []
    manager         = torch.multiprocessing.Manager()
    q               = manager.Queue()
    graph_list      = np.array_split(prediction_graphs,n_workers)
    slaves          = Spawn_Slaves(n_workers, graph_list, q)
    count           = torch.tensor([0]).to(device)
    dead_check      = False
    model.eval()
    with torch.no_grad():
        for mini_batch in range(0,pred_mini_batches):
            data            = GrabBatch(q)
            prediction      = model(data)
            truth           = data.y[:,0].unsqueeze(1).detach().cpu().numpy()
            pred_events.extend(data.event_no.detach().cpu().numpy())
            predictions.extend(prediction.detach().cpu().numpy())
            truths.extend(truth)
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
        truths          = pd.DataFrame(truths)
        predictions     = pd.DataFrame(predictions)
        pred_events     = pd.DataFrame(pred_events)
        
        result          = pd.concat([pred_events,truths, predictions],axis = 1)
        result.columns  = ['event_no','E','E_pred']
        result.to_csv(path + '\\' + 'results_%s.csv'%o)
        
        os.makedirs(r'X:\speciale\results\%s\%s'%(handle,baseline),exist_ok = True)
        result.to_csv(r'X:\speciale\results\%s\%s\results.csv'%(handle,baseline))
        torch.save(model.state_dict,r'X:\speciale\results\%s\%s\state_dict.pkl'%(handle,baseline))
        torch.save(model.parameters(),r'X:\speciale\results\%s\%s\parameters.pkl'%(handle,baseline))
        print('Result Saved at: \n \
              reports: %s \
              archive: %s '%(path + '\\' + 'results_%s.csv'%o, r'X:\speciale\results\%s\%s'%(handle,baseline)))
        
        
        
        
        
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
    path = r'X:\speciale\data\graphs'
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
    
    handle = 'dev_numu_train_upgrade_step4_2020_00\event_only_shuffled_input(0,1)_target-_padded_2mio'
    base = 'tgcnv2-baseline-upgrade'
    scheduler = 'LinearSchedule'
    
    
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
    graphs_train             = np.array(GrabGraphs(path = r'X:\speciale\data\graphs\%s\train'%handle))
    graphs_valid             = GrabGraphs(r'X:\speciale\data\graphs\%s\valid'%handle)
    mini_batches,mini_batches_valid             = Calculate_Minibatches(graphs_train,graphs_valid, handle, batch_size)
    max_iter                 = mini_batches*n_epochs
    steps_up                 =  0.1*max_iter
    steps_down               = max_iter  - steps_up  
    #lr_list                 = Get_lr_list(lr,max_lr,end_lr,steps_up,steps_down,batch_size,max_iter,device)
    lr_list                  =  torch.tensor(LinearSchedule(lr,max_lr,end_lr,steps_up = int(mini_batches/2), steps_down = mini_batches*n_epochs - int(mini_batches/2))).to(device)
    es                       = EarlyStopping(patience=patience)
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
    
    ### TRAINING
    writer =  SummaryWriter(log_dir= "logs/dynedge/tmp/%s"%o)
    trained_model,loss_history = Train(Net(),graphs_train,device,lr,lr_list,n_workers,batch_size,loss_decimals,graphs_valid,mini_batches_valid)
    
   # pd.DataFrame(loss_history).to_csv(r'J:\speciale\results\runs\results\event_only\%s_loss.csv'%o)
    writer.close()
    ### Prediction

    Predict(trained_model,graphs_valid,n_workers,mini_batches_valid,batch_size,o,handle,base)
    
    
    ### LR FINDER
    #loss,lr = LRFinder(10,Net(),[1e-7,1e-1],NMAELoss,graphs_train,batch_size,mini_batches,n_workers)
   #
    #median = np.median(np.array(loss),0)
    #mean = np.mean(np.array(loss),0)
    #plt.plot(lr,median)
    #plt.plot(lr,mean)
    #plt.legend(['median','mean'])
    playsound(r'X:\speciale\resources\warpcorecollapse.mp3')