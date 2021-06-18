import pandas as pd
import numpy as np
import math

test = pd.read_csv("J:\\speciale\\results\\40-48-node-split\\type1_seq.csv")

def CalculateInertiaTensor(df):
    events = df['event_no'].unique()
    result = list()
    for event in events:
        I = np.zeros([4,4])
        x = df['dom_x'][df['event_no'] == event]
        y = df['dom_y'][df['event_no'] == event]
        z = df['dom_z'][df['event_no'] == event]
        t = df['dom_time'][df['event_no'] == event]
        charge = df['dom_charge'][df['event_no'] == event]
        r = np.sqrt(x**2 + y**2 + z**2 +t**2)
        
        pos = list()
        pos.append(t)
        pos.append(x)
        pos.append(y)
        pos.append(z)
        for i in range(0,4):
            for j in range(0,4):
                if i == j:
                    I[i,j] = (charge*(r**2 - pos[i]*pos[j])).sum()
                else:
                    I[i,j] = (charge*(-pos[i]*pos[j])).sum()
        result.append(np.linalg.eig(I)[0])
    return result


b = CalculateInertiaTensor(test)