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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import torch_geometric
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GatedGraphConv
import matplotlib.pyplot as plt
from torch.nn import MSELoss
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", type=int, required=True)
parser.add_argument("-tag", "--tag", type=str, required=True)
parser.add_argument("-graphs","--graphs",type = str,required = True)
parser.add_argument("-device","--device",type = str,required = True)
args = parser.parse_args()
device = torch.device(args.device)



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

def NMAELoss(output,target,w,scalers):
    eps = 1e-3
    k            = torch.abs(output[:,4])
    #azimuth = torch.tensor(scalers['truth']['azimuth'].inverse_transform(np.array(target[:,4].cpu()).reshape(-1,1)),dtype=torch.float).to(device)
    #zenith  = torch.tensor(scalers['truth']['zenith'].inverse_transform(np.array(target[:,5].cpu()).reshape(-1,1)),dtype=torch.float).to(device)
    azimuth = target[:,4]
    zenith  = target[:,5]
    
    sin_ø = torch.sin(zenith)          #output1
    sin_ø2 = torch.sin(azimuth)        #output2 
    cos_ø  = torch.cos(zenith)         #output3
    cos_ø2 = torch.cos(azimuth)        #output4 
    
    norm_u = 1/torch.sqrt(torch.tensor(2,dtype = torch.float))
   
    u_1 = norm_u*cos_ø2*sin_ø
    u_2 = norm_u*sin_ø2*sin_ø
    u_3 = norm_u*cos_ø

    x_1 = output[:,3]*output[:,0]
    x_2 = output[:,1]*output[:,0]
    x_3 = output[:,2]

    norm_x  = 1/torch.sqrt((x_1)**2 + (x_2)**2 + (x_3)**2)
    
    x_1 = norm_x*x_1
    x_2 = norm_x*x_2
    x_3 = norm_x*x_3

    dotprod = u_1*x_1 + u_2*x_2 + u_3*x_3

    logc_3 = - torch.log(k) + k + torch.log(1 - torch.exp(-2*k))
    
    loss = torch.mean(-k*dotprod + logc_3)
    
    return loss


    

start = time.time()                                                            

class Net(torch.nn.Module):                                                     
    def __init__(self, k):                                                                                   
        super(Net, self).__init__()
        c = 3
        l1, l2, l3, l4, l5,l6,l7 = 8,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,5
        self.k = k
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)
        self.conv_max = EdgeConv(self.nn_conv1,aggr = 'max')
        self.conv_mean = EdgeConv(self.nn_conv1,aggr = 'mean')
        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

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
        
        x[:,0] = self.tanh(x[:,0])
        x[:,1] = self.tanh(x[:,1])
        x[:,2] = self.tanh(x[:,2])
        x[:,3] = self.tanh(x[:,3])
        

        

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
    print('All task requests sent')
    return slaves

def Create_Slaves(n_workers,graphs_valid,manager,batch_size):
    if(n_workers > len(graphs_valid)):
        graph_list = np.array_split(graphs_valid,len(graphs_valid))
    else:
        graph_list = np.array_split(graphs_valid,n_workers)
    q_val = manager.Queue()
    val_slaves = Spawn_Slaves(n_workers,graph_list,q_val,batch_size)
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
        slaves = Spawn_Slaves(n_workers, graph_list, q,batch_size)
        for k in range(0,mini_batches):                                               
            data_train                      = GrabBatch(q)
            #data_train.y                    = data_train.y[:,0].unsqueeze(1)#torch.tensor(scaler.inverse_transform(np.array(data_train.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device) #data_train.y[:,0].unsqueeze(1)                                         
            model_LR.train()                                                               
            w                               = 1 
            optimizer.zero_grad()                                                   
            out                         = model(data_train)
            if(torch.sum(torch.isnan(out)) != 0):
                print('NAN ENCOUNTERED AT : %s / %s'%(k,mini_batches))
                                                            
            loss                            = loss_func(out, data_train.y,w)                             
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
def Train(model_to_train,graphs_train,device,lr,lr_list,n_workers,batch_size,loss_decimals,graphs_valid,mini_batches_valid,scalers):                                                                                
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
                slaves = Spawn_Slaves(n_workers, graph_list, q,batch_size)
            for k in range(0,mini_batches):                                               
                data_train                      = GrabBatch(q)
                with torch.enable_grad():
                    data_train.y                    = data_train.y#torch.tensor(scaler.inverse_transform(np.array(data_train.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device) #data_train.y[:,0].unsqueeze(1)                                         
                    model.train()                                                               
                    w                               = data_train.hw
                    optimizer.zero_grad()                                                   
                    out                         = model(data_train)
                    if(torch.sum(torch.isnan(out)) != 0):
                        print('NAN ENCOUNTERED AT : %s / %s'%(k,mini_batches))
                                                                    
                    loss                            = loss_func(out, data_train.y,w,scalers)                             
                    #loss = torch.sum(torch.abs(out - data_train.y.float())*w)
                    loss.backward()                                                         
                    optimizer.step()
                
                optimizer.param_groups[0]['lr'] = lr_list[count].item()
                count                           +=1
                loss_acc                        +=loss
                dead_check = Slave_Watcher(slaves, dead_check, k, mini_batches)
                if(dead_check  == True and transition_val == False):
                    q_val,val_slaves = Create_Slaves(n_workers, graphs_valid, manager,batch_size)
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
                q_val,val_slaves =  Create_Slaves(n_workers, graphs_valid, manager,batch_size)
            print('entering validation')
            with torch.no_grad():
                val_loss,q,slaves = Validate(model,graphs_valid,graphs_train,batch_size,o,
                                             q_val,mini_batches_valid,val_slaves,n_workers,mini_batches,manager,epoch,n_epochs,scalers,save = False)
            if epoch == 0:
                deltatime                        = (time.time() - start)/60
            #### TENSORBOARD
            writer.add_scalar('val_loss_acc', val_loss.item(),epoch)
            writer.add_scalar('train_loss_acc',loss_acc.item(),epoch)
            writer.flush()
            ####
            print('EPOCH: %s / %s || %s / %s min || LR: %s || Loss: %s || Val Loss: %s' %(epoch,n_epochs,(time.time() - start)/60,n_epochs*deltatime,optimizer.param_groups[0]['lr'],round(loss_acc.item()/mini_batches,loss_decimals),round(val_loss.item()/mini_batches_valid,loss_decimals)))
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
        del data_train
    return model,loss_pr_epoch
def Validate(trained_model,graphs_valid,graphs_train,batch_size,
             o,q,mini_batches_valid,val_slaves,n_workers,mini_batches,manager,train_epoch,n_epochs,scalers,save):
    print('in validation')
    model = trained_model
    loss_func = NMAELoss
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
            data_pred.y = data_pred.y #torch.tensor(scaler.inverse_transform(np.array(data_pred.y[:,0]).reshape(-1,1)),dtype=torch.float).to(device)#data_pred.y[:,0].unsqueeze(1)

            pred = model(data_pred)
            w                               = data_pred.hw 
            loss = loss_func(pred,data_pred.y,w,scalers)
            #loss = torch.sum(torch.abs(pred - data_pred.y.float())*w)
            if torch.isnan(loss):
                print('Warning: NaN encountered at %s / %s. Skipping this sample!'%(i,mini_batches_valid) )
            else:    
                acc_loss += loss                                                          # PREDICTION AND CALCULATION OF NMAE-SCORE                                                        
            dead_check = Slave_Watcher(val_slaves, dead_check, i, mini_batches_valid)
            if(dead_check  == True and transition_val == False and train_epoch < n_epochs - 1):
                q_train,train_slaves = Create_Slaves(n_workers, graphs_train, manager,batch_size)
                transition_val = True

            
        if( dead_check == False or q_train == None):
            print('Validation Slaves Still Alive. Getting ChainSaw..')
            for val_slave in val_slaves:
                val_slave.terminate()
            q_train,train_slaves = Create_Slaves(n_workers, graphs_train, manager,batch_size)
     
        val_loss = acc_loss
        return val_loss,q_train,train_slaves
def Predict(model,prediction_graphs,n_workers,pred_mini_batches,batch_size,o,handle,baseline,scalers):
    path     = '/groups/hep/pcs557/speciale/results/runs/results/event_only'
    os.makedirs(path,exist_ok = True)
    print('PREDICTING: \n \
          model   : %s \n \
          n_events: %s' %(baseline,pred_mini_batches*batch_size))
    os.makedirs('/groups/hep/pcs557/speciale/results/%s/%s'%(handle,baseline),exist_ok = True)
    torch.save(model,'/groups/hep/pcs557/speciale/results/%s/%s/model.pkl'%(handle,baseline))
    predictions     = []
    truths          = []
    pred_events     = []
    manager         = torch.multiprocessing.Manager()
    q               = manager.Queue()
    graph_list      = np.array_split(prediction_graphs,n_workers)
    slaves          = Spawn_Slaves(n_workers, graph_list, q,batch_size)
    count           = torch.tensor([0]).to(device)
    dead_check      = False
    E               = []
    sigma           = []
    model.eval()
    with torch.no_grad():
        for mini_batch in range(0,pred_mini_batches):
            data            = GrabBatch(q)
            prediction      = model(data)
            truth           = data.y[:,1:6].detach().cpu().numpy()
            pred_events.extend(data.event_no.detach().cpu().numpy())
            predictions.extend(prediction.detach().cpu().numpy())#torch.atan2(prediction[:,0],prediction[:,1]).detach().cpu().numpy())
            truths.extend(truth)
            E.extend(data.y[:,0].unsqueeze(1).detach().cpu().numpy())
            #sigma.extend(prediction[:,4].unsqueeze(1).detach().cpu().numpy())
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
        #sigma           = pd.DataFrame(sigma)
        
        result          = pd.concat([pred_events,E,truths, predictions],axis = 1)
        print(result.shape)
        print(predictions.shape)
        print(truths.shape)
        #result          = pd.concat([pred_events, predictions,sigma],axis = 1)
        result.columns  = ['event_no','energy_log10','dir_x','dir_y','dir_z','azimuth','zenith',
                           'sin_zenith','sin_azimuth', 'cos_zenith', 'cos_azimuth', 'k']
        #result.columns  = ['event_no','zenith_pred','zenith_pred_k']
        result.to_csv(path + '/' + 'results_%s.csv'%o)
        
        os.makedirs('/groups/hep/pcs557/speciale/results/%s/%s'%(handle,baseline),exist_ok = True)
        result.to_csv('/groups/hep/pcs557/speciale/results/%s/%s/results.csv'%(handle,baseline))
        #torch.save(model.state_dict,r'J:\speciale\results\%s\%s\state_dict.pkl'%(handle,baseline))
        #torch.save(model.parameters(),r'J:\speciale\results\%s\%s\parameters.pkl'%(handle,baseline))
        print('Result Saved at: \n \
              reports: %s \
              archive: %s '%(path + '/' + 'results_%s.csv'%o, '/groups/hep/pcs557/speciale/results/%s/%s'%(handle,baseline)))
        
        
        
        
        
def GrabGraphs(path):
    graphs = list()
    for file in os.listdir(path):
        graphs.append(path + '//' + file)
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
    path = '/groups/hep/pcs557/speciale/data/graphs'
    check = os.path.isfile(path + '/' + handle + '/minibatch/minibatch_GRU.csv')
    if(check == False):
        print('Mini-batch information for training not found. Generating..')
        os.makedirs(path + '/' + handle + '/minibatch',exist_ok = True)
        mini_batches = 0
        for k in range(0,len(graphs_train)):
            print('%s / %s'%(k,len(graphs_train)))
            data_list_train = torch.load(graphs_train[k])                                                                                                                                                  
            loader = DataLoader(data_list_train, batch_size = batch_size,drop_last = True)
            mini_batches +=len(loader)
        pd.DataFrame([mini_batches]).to_csv(path + '/' + handle + '/minibatch/minibatch_GRU.csv',)
    else:
        print('Mini-batch information for training found at \n %s'%(path + '/' + handle + '/minibatch/minibatch_GRU.csv'))
        mini_batches = pd.read_csv(path + '/' + handle + r'/minibatch/minibatch_GRU.csv').values[0][1]
        
    check = os.path.isfile(path + '/' + handle + '/minibatch/minibatch_valid_GRU.csv')
    if(check == False):
        print('Mini-batch information for validation not found. Generating..')
        os.makedirs(path + '/' + handle + r'/minibatch',exist_ok = True)
        mini_batches_valid = 0
        for k in range(0,len(graphs_valid)):
            print('%s / %s'%(k,len(graphs_valid)))
            data_list_valid = torch.load(graphs_valid[k])                                                                                                                                                  
            loader = DataLoader(data_list_valid, batch_size = batch_size,drop_last = True)
            mini_batches_valid +=len(loader)
        pd.DataFrame([mini_batches_valid]).to_csv(path + '/' + handle + '/minibatch/minibatch_valid_GRU.csv',)
    else:
        print('Mini-batch information for validation found at \n %s'%(path + '/' + handle + '/minibatch/minibatch_valid_GRU.csv'))
        mini_batches_valid = pd.read_csv(path + '/' + handle + '/minibatch/minibatch_valid_GRU.csv').values[0][1]
    return mini_batches,mini_batches_valid
        
        
###########################
#       CONFIGURATION     #
###########################
if __name__ == '__main__':

    handle = args.graphs #'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_1mio' #'dev_level7_oscNext_IC86_003/event_only_level7_all_oscNext_IC86_003' #' 
    base = 'dynedge-protov2-zenith-k=%s-c3not3-%s'%(args.k, args.tag)
    scheduler = 'LinearSchedule'    
         
    batch_size      = 1048 # 
    n_workers       = 30                                                          
    n_reps          = 1                                                                                                                                    
    lr              = 1e-4#7e-5#7e-4
    max_lr          = 2.5e-3 #2.5e-3#2.5e-4#2.5e-3
    end_lr          = 1e-5  #1e-5#7e-5#7e-4                                                               
    n_epochs        = 50 #1990
    patience        = 5
    loss_decimals   = 10
    
    #weights                  = pd.read_csv("~/speciale/data/graphs/dev_numu_train_l5_retro_001/event_only_aggr_test_SRT_energy_scaled_new_weights/weights/weights.csv")
    scalers                  = pd.read_pickle('/groups/hep/pcs557/speciale/data/raw/dev_level7_mu_e_tau_oscweight_newfeats/meta/transformers.pkl')
    graphs_train             = np.array(GrabGraphs(path = '/groups/hep/pcs557/speciale/data/graphs/%s/train'%handle))
    graphs_valid             = GrabGraphs('/groups/hep/pcs557/speciale/data/graphs/%s/valid'%handle)
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
    trained_model,loss_history = Train(Net(args.k).to(device),graphs_train,device,lr,lr_list,n_workers,batch_size,loss_decimals,graphs_valid,mini_batches_valid,scalers)
    #trained_model = torch.load('/groups/hep/pcs557/speciale/results/dev_level7_mu_e_tau_oscweight_newfeats/everything_wtest/dynedge-protov2-zenith-k=8-c3not3-w_test_val/model.pkl', map_location='cpu').to(device)
    #pd.DataFrame(loss_history).to_csv(r'J:\speciale\results\runs\results\event_only\%s_loss.csv'%o)
    writer.close()
    ### Prediction

    Predict(trained_model,graphs_valid,n_workers,mini_batches_valid,batch_size,o,handle,base,scalers)
    
    
    ### LR FINDER
    #n_reps = 3
    #loss_list = []
    #lr_list = []
    #for n in range(0,n_reps):
    #    loss,lr = LRFinder(1,Net(args.k),[1e-8,1e-4],NMAELoss,graphs_train,batch_size,mini_batches,n_workers)
    #    loss_list.append(loss)
    #    lr_list.append(lr)
    #for l in range(0,len(loss_list)):
    #    fig = plt.figure()
    #    plt.plot(np.array(lr_list[l]), np.array(loss_list[l]).squeeze())
    #    fig.savefig('%s-%s.pdf'%(base,l))
    #    plt.show()
    #median = np.median(np.array(loss),0)
    #mean = np.mean(np.array(loss),0)
    #plt.plot(lr,median)
    #plt.plot(lr,mean)
    #plt.legend(['median','mean'])
    #playsound(r'J:\speciale\resources\warpcorecollapse.mp3')

