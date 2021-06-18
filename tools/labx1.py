import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from multiprocessing import Pool
import multiprocessing
import time
import os

def Get_lr_list(lr,max_lr,end_lr,steps_up,steps_down,batch_size,max_iter, mode):
    lr,max_lr,end_lr,steps_up,steps_down,batch_size,max_iter
    lr_factor = list()
    if mode == 'normal':
        for k in range(0,int(max_iter)):
           lr_factor.append(lr_watcher(lr,max_lr,end_lr,steps_up,steps_down,batch_size,schedule = 'inverse',iterations_completed = k+1).get_factor())
        
        lr_list = np.array(lr_factor)*lr
    if mode == 'experimental':
        
        for k in range(0,int(max_iter)):
            lr_factor.append(lr_watcher_test(lr,max_lr,end_lr,steps_up,steps_down,k))
        
        lr_list = np.array(lr_factor)*lr 
    return lr_list

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
def lr_watcher_test(start_lr,max_lr,min_lr,steps_up,steps_down,step):
    
    if step <= steps_up:
        k = (max_lr/start_lr)**(1/steps_up)*step*start_lr
    if step > steps_up:
        k = (max_lr/start_lr)*steps_down*(min_lr/(min_lr + (max_lr - min_lr)*(step - steps_up)))
    return k
    
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
        self._steps_up = n_rise
        self._steps_down = steps_down
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

def weight(x,x_threshold,type):
    if type == 'low':
        if x<x_threshold:
            return 1
    
        if x>= x_threshold:
            return 1*(1/(1+ x-x_threshold))
    if type == 'high':
        if(x> x_threshold):
            return 1
        if(x<= x_threshold):
            return 1*(1/(1 + x_threshold-x))

data = np.arange(0,4,0.1)

x_min = 0.5
lw = []
for d in data:
    lw.append(weight(d,x_min,'low'))
weights  = np.array(lw)
diff = 1 - np.mean(weights)
lw  = np.array(weights) + diff

x_min = 2.5
hw = []
count = 0
for d in data:
    hw.append(weight(d,x_min,'high'))
    count +=1
    if count  == 25:
        print(weight(d,x_min,'high'))
        
weights  = np.array(hw)
diff = 1 - np.mean(weights)
hw  = np.array(weights) + diff


plt.plot(data,lw)
plt.plot(data,hw)
plt.legend(['$w(x,0.5)_{low}$','$w(x,2.5)_{high}$'])
plt.ylabel('Weight', size = 20)
plt.xlabel('x',size = 20)
plt.title('High and Low Ensemble Weights',size = 20)

#### LR SCHEDULES

batch_size      = 2*1024
n_workers       = 5                                                          
n_reps          = 1                                                                                                                                    
lr              = 7e-4
max_lr          = 2.5e-3
end_lr          = 7e-5                                                                 
n_epochs        = 100 #1990

max_iter                 = 360*n_epochs
steps_up                 =  int(0.1*max_iter)
steps_down               = (max_iter  - steps_up)

lr_list_inverse          = Get_lr_list(lr,max_lr,end_lr,steps_up,steps_down,batch_size,max_iter,mode = 'normal')
lr_list_inverse2          = Get_lr_list(lr,max_lr,end_lr,steps_up,steps_down,batch_size,max_iter,mode = 'experimental')
lr_list                  =  LinearSchedule(lr,max_lr,end_lr,steps_up = steps_up, steps_down = steps_down)

indicator_max = np.repeat(steps_up,100)
y = np.linspace(0,max_lr,100)
indicator_0 = np.repeat(0,100)


plt.close()
plt.plot(range(0,len(lr_list)),lr_list)
plt.plot(range(0,len(lr_list_inverse)),lr_list_inverse)
plt.plot(range(0,len(lr_list_inverse2)),lr_list_inverse2)
plt.scatter(0,lr,color = 'black')
plt.scatter(steps_up,max_lr,color = 'black')

plt.text(3, lr -1/4*lr, '$LR_{init.}$', style='italic', fontsize=12)
plt.text(steps_up + 2000, max_lr, '$LR_{max}$', style='italic', fontsize=12)

plt.legend(['Linear','Inverse'])
plt.ylabel('LR',size = 20)
plt.xlabel('Optimization Steps',size = 20)
plt.title('LR Schedules', size = 20)














