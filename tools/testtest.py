import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import keyboard
from matplotlib.cm import get_cmap

cmap = get_cmap('tab20b')

events = pd.read_csv(r'X:\speciale\data\export\Upgrade Cleaning\selections\martin_selection.csv')['event_no'].reset_index(drop = True)
mc_db = r'X:\speciale\data\raw\dev_upgrade_train_step4_001\data\dev_upgrade_train_step4_001.db'


scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_upgrade_train_step4_001\meta\transformers.pkl')


x_scaler = scalers['features']['dom_x']
y_scaler = scalers['features']['dom_y']
z_scaler = scalers['features']['dom_z']
pmt_x_scaler = scalers['features']['dom_x']
pmt_y_scaler = scalers['features']['dom_y']
pmt_z_scaler = scalers['features']['dom_z']
time_scaler = scalers['features']['time']


with sqlite3.connect(mc_db) as con:
    query = 'select event_no from truth where event_no not in %s'%(str(tuple(events)))
    data_main = pd.read_sql(query,con)



bad_event = data_main['event_no'].reset_index(drop = True)

counter = 0
checker = True
fig,ax = plt.subplots(2)
while checker == True:
    if keyboard.is_pressed('right'):
        ax[0].clear()
        ax[1].clear()
        counter = counter + 1
        if counter > (len(data_main) -1):
            counter = (len(data_main) -1)
        bad_events = bad_event[counter]
        print(bad_events)
        
        pause = 0.5
        
        with sqlite3.connect(mc_db) as con:
            query = 'select * from features where event_no = %s and SRTInIcePulses = 1'%(bad_events)
            data = pd.read_sql(query,con)
            
        
        
        #data['dom_x'] = 1000 + x_scaler.inverse_transform(np.array(data['dom_x']).reshape(-1,1))
        #data['dom_y'] = 1000 + y_scaler.inverse_transform(np.array(data['dom_y']).reshape(-1,1)) 
        #data['dom_z'] = 1000 + z_scaler.inverse_transform(np.array(data['dom_z']).reshape(-1,1))  
        #data['pmt_x'] = 1000 + pmt_x_scaler.inverse_transform(np.array(data['pmt_x']).reshape(-1,1))
        #data['pmt_y'] = 1000 + pmt_y_scaler.inverse_transform(np.array(data['pmt_y']).reshape(-1,1)) 
        #data['pmt_z'] = 1000 + pmt_z_scaler.inverse_transform(np.array(data['pmt_z']).reshape(-1,1))
        #data['time'] = time_scaler.inverse_transform(np.array(data['time']).reshape(-1,1))   
        first_event = data
                
        
        plt.pause(0.1)
        plt.ion()
        index = first_event['pmt_type']==130
        pmts = pd.unique(first_event['pmt'][index])
        plot_data = first_event.loc[index,:].reset_index(drop = True)
        for k in range(0,len( plot_data)):
            
            ax[0].scatter(plot_data['time'][k],
                       plot_data['dom'][k] + plot_data['string'][k],  
                       color = cmap(np.where(pmts == plot_data['pmt'][k])[0][0]/len(pmts)))
        ax[0].set_ylabel('mDOM-ID',size = 20)    
        ax[0].set_title('mDOM Activations', size = 20)
        ax[0].set_xlabel('Rel. Trigger Time [ns]', size = 20)
        index = first_event['pmt_type']==20
        ax[1].scatter(first_event['time'][index],first_event['dom'][index])
        ax[1].set_title('DOM')
        plt.pause(0.1)
        plt.ion()
        plt.show()
        fig.canvas.draw()
        
        #counter = counter + 1
        
    if keyboard.is_pressed('left'):
        ax[0].clear()
        ax[1].clear()
        counter = counter - 1
        if counter < 0:
            counter = 0
        bad_events = bad_event[counter]
        print(bad_events)
        
        pause = 0.5
        
        with sqlite3.connect(mc_db) as con:
            query = 'select * from features where event_no = %s and SRTInIcePulses = 1 '%(bad_events)
            data = pd.read_sql(query,con)
            
        
        
        data['dom_x'] = 1000 + x_scaler.inverse_transform(np.array(data['dom_x']).reshape(-1,1))
        data['dom_y'] = 1000 + y_scaler.inverse_transform(np.array(data['dom_y']).reshape(-1,1)) 
        data['dom_z'] = 1000 + z_scaler.inverse_transform(np.array(data['dom_z']).reshape(-1,1))  
        data['pmt_x'] = 1000 + pmt_x_scaler.inverse_transform(np.array(data['pmt_x']).reshape(-1,1))
        data['pmt_y'] = 1000 + pmt_y_scaler.inverse_transform(np.array(data['pmt_y']).reshape(-1,1)) 
        data['pmt_z'] = 1000 + pmt_z_scaler.inverse_transform(np.array(data['pmt_z']).reshape(-1,1))
        data['time'] = time_scaler.inverse_transform(np.array(data['time']).reshape(-1,1))   
        first_event = data
                
        
        plt.pause(0.1)
        plt.ion()
        index = first_event['pmt_type']==130
        pmts = pd.unique(first_event['pmt'][index])
        plot_data = first_event.loc[index,:].reset_index(drop = True)
        for k in range(0,len( plot_data)):
            
            ax[0].scatter(plot_data['time'][k],
                       plot_data['dom'][k], 
                       color = cmap(np.where(pmts == plot_data['pmt'][k])[0][0]/len(pmts)))
            
            ax[0].set_title('mDOM')
        index = first_event['pmt_type']==20
        ax[1].scatter(first_event['time'][index],first_event['dom'][index])
        ax[1].set_title('DOM')
        plt.pause(0.1)
        plt.ion()
        plt.show()
        fig.canvas.draw()
    if keyboard.is_pressed('esc'):
        checker = False
        
    


