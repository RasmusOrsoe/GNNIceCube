import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path = r'J:\speciale\results\runs\results\event_only\lrfinder'
val_loss = []
epochs = []
for file in os.listdir(path):
    if(file[-10:] == 'result.csv'):
        result = pd.read_csv(path + '\\' +file)
        loss_acc = sum(abs(result['E'] - result['E_pred']))
        val_loss.append(loss_acc)
        check_epoch = file[71:73]
        if check_epoch[1] == '_':
            epoch = int(check_epoch[0])
        else:
            epoch = int(check_epoch)
        epochs.append(epoch)
    
data = pd.DataFrame(epochs)
data['loss'] = val_loss
data.columns = ['epoch','loss']
data = data.sort_values('epoch')

plt.plot(data['epoch'],data['loss'])