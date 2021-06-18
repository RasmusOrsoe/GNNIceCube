import os
import random 
import numpy as np


def MakeTestTrain(path,trainseq):
    files = os.listdir(path)
    csv_files = []
    graphs = []
    for file in files:
        if '.csv' in file:
            csv_files.append(file)
        if '.pkl' in file:
            graphs.append(file)
            
    random.shuffle(csv_files)
    csv_file_index = np.arange(0,len(csv_files),1)
    
    chunks = np.array_split(csv_file_index,10)
    train = chunks[0:6]
    valid = chunks[6:8]
    test  = chunks[8:10]
    
    os.makedirs(path + '\\events')
    os.makedirs(path + '\\events\\train')
    os.makedirs(path + '\\events\\valid')
    os.makedirs(path + '\\events\\test')
    
    os.makedirs(path + '\\train')
    os.makedirs(path + '\\valid')
    os.makedirs(path + '\\test')
    
    for i in range(0,len(train)):
        print('TRAIN: %s / %s'%(i,len(train)))
        for file_index in train[i]:
            csv_name = csv_files[file_index].split('_')[2].split('.')[0]
            for graph in graphs:
                graph_name = graph.split('_')[2].split('.')[0]
                
                if graph_name == csv_name:
                    os.rename(path + '\\' + csv_files[file_index],path + '\\events\\train\\' + csv_files[file_index])
                    os.rename(path + '\\' + graph,path + '\\train\\' + graph)
                
    for i in range(0,len(valid)):
        print('VALID: %s/%s'%(i,len(valid)))
        for file_index in valid[i]:
            csv_name = csv_files[file_index].split('_')[2].split('.')[0]
            for graph in graphs:
                graph_name = graph.split('_')[2].split('.')[0]
                
                if graph_name == csv_name:
                    os.rename(path + '\\' + csv_files[file_index],path + '\\events\\valid\\' + csv_files[file_index])
                    os.rename(path + '\\' + graph,path + '\\valid\\' + graph)
                                
    for i in range(0,len(test)):
        print('TEST: %s/%s'%(i,len(test)))
        for file_index in test[i]:
            csv_name = csv_files[file_index].split('_')[2].split('.')[0]
            for graph in graphs:
                graph_name = graph.split('_')[2].split('.')[0]
                
                if graph_name == csv_name:
                    os.rename(path + '\\' + csv_files[file_index],path + '\\events\\test\\' + csv_files[file_index])
                    os.rename(path + '\\' + graph,path + '\\test\\' + graph)
                        
        
    return



path = r'X:\speciale\data\graphs\dev_level7_mu_e_tau_oscweight_newfeats\track_cascadev2'

graphs = MakeTestTrain(path,1)