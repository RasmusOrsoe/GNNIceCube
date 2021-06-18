import pandas as pd
import numpy as np
import os
import sqlite3
from sklearn.preprocessing import RobustScaler
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def GrabGCD(path):
    print('GRABBING GEO-SPATIAL DATA')
    files = os.listdir(path)
    for file in files:
        if file.endswith('.pkl'):
            gcd = pd.read_pickle(path + '\\' + file)
    return gcd['geo']

def parse_args(description=__doc__):
    """Parse command line args"""
    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--array_path', type=str, required=True,
        help='The path in which the array is saved.',
    )
    parser.add_argument(
        '--key', type=str, required=True,
        help='The key for the array. e.g. SplitInIcePulses ',
    )
    parser.add_argument(
        '--db_name', type=str, required=True,
        help='Name of database - no extension please. E.g: MyNewDataBase',
    )
    parser.add_argument(
        '--gcd_path', metavar='GCD_PATH', dest='gcd_path', type=str,
        required=True,
        help='Input GCD file.'
    )
    parser.add_argument(
        '--outdir', type=str, required=True,
        help='Directory into which to save .db file and transform.pkl',
    )
    return parser.parse_args()
def CreateDataBase(array_path,key,db_name,gcd_path,outdir):
    #array_path            = r'X:\speciale\hep\arrays_from_hep\arrays'
    #gcd_path              = r'X:\speciale\hep\gcd\gcd_array'
    #key                   = 'SplitInIcePulses'
    #outdir                = r'X:\speciale\hep\arrays_from_hep\output'
    #db_name               = 'test-db'
    
    path                  = array_path + '\\' + key
    print('LOADING %s FEATURE ARRAY...'%key)
    data                  = np.load(path + '\data.npy')
    print('LOADING %s FEATURE INDEX...'%key)
    data_index            = np.load(path + '\index.npy')
    
    
    truth_key             = 'MCInIcePrimary'
    print('LOADING %s TRUTH ARRAY'%truth_key) 
    path_truth            = array_path + '\\' + truth_key
    truth                 = np.load(path_truth + '\\' + 'data.npy')
    
    ####################################
    #                                  #
    #      LEGACY TRUTH VARIABLES      #
    #                                  #
    ####################################
    
    print('EXTRACTING TRUTH VALUES..')
    
    pid             = truth['pdg_encoding']
    position_x      = truth['pos']['x']
    position_y      = truth['pos']['y']
    position_z      = truth['pos']['z']
    zenith          = truth['dir']['zenith']
    azimuth         = truth['dir']['azimuth']
    time            = truth['time']
    energy_log10    = truth['energy']
    track_length    = truth['length']
    event_no_truth  = np.arange(1,len(pid) + 1)
    
    #feats = str('event_no,x,y,z,time,charge_log10')
    #truths = str('event_no,energy_log10,time,vertex_x,vertex_y,vertex_z,direction_x,direction_y,direction_z,azimuth,zenith,pid')                                                       #                                                      #
    
    
    truth           = pd.concat([pd.DataFrame(event_no_truth[0:1000]),
                                pd.DataFrame(energy_log10[0:1000]),
                                pd.DataFrame(time[0:1000]),
                                pd.DataFrame(position_x[0:1000]),
                                pd.DataFrame(position_y[0:1000]),
                                pd.DataFrame(position_z[0:1000]),
                                pd.DataFrame(azimuth[0:1000]),
                                pd.DataFrame(zenith[0:1000]),
                                pd.DataFrame(pid[0:1000]),
                                pd.DataFrame(track_length[0:1000])]
                                ,axis = 1)
    truth.columns   = ['event_no',
                    'energy_log10',
                    'time',
                    'position_x',
                    'position_y',
                    'position_z',
                    'azimuth',
                    'zenith',
                    'pid',
                    'muon_track_length']
    
    ####################################
    #                                  #
    #     LEGACY FEATURES VARIABLES    #
    #                                  #
    ####################################
    print('EXTRACTING FEATURES ...')
    geo                 = GrabGCD(gcd_path)
    hits_idx            = data_index
    hits                = data
    single_hits         = np.empty(hits.shape + (5,))
    event_no            = np.empty((len(hits),))
    string_idx          = hits['key']['string'] - 1
    om_idx              = hits['key']['om'] - 1
    single_hits[:, 0:3] = geo[string_idx, om_idx]
    single_hits[:, 3]   = hits['pulse']['time']
    single_hits[:, 4]   = hits['pulse']['charge']
    ## Get charge per event and string
    print('EXTRACTING GEO-SPATIAL DOM DATA')
    for i in range(len(hits_idx)):
        this_idx = hits_idx[i]
        event_no[this_idx['start'] : this_idx['stop']] = i + 1
        
    features            = pd.concat([pd.DataFrame(event_no)[0:1000],
                                      pd.DataFrame(single_hits[0:1000])]
                                      ,axis = 1)
    features.columns    = ['event_no',
                            'dom_x',
                            'dom_y',
                            'dom_z',
                            'dom_time',
                            'charge_log10']
    
    ####################################
    #                                  #
    #          PREPROCESSING           #
    #                                  #
    ####################################    
    
    feature_keys        = features.columns
    feature_keys        = feature_keys[feature_keys != 'event_no']
    
    truth_keys          = truth.columns
    truth_keys          = truth_keys[truth_keys != 'event_no']
    truth_keys          = truth_keys[truth_keys != 'pid']
    
    transformer_dict    = {'input':{}, 'truth':{}}
    
    for key in feature_keys:
        print('FITTING %s TRANSFORMER' %key)
        scaler          = RobustScaler()
        scaler          = scaler.fit(np.array(features[key]).reshape(-1,1))
        features[key]   = scaler.transform(np.array(features[key]).reshape(-1,1))
        transformer_dict['input'][key] = scaler
    
    for key in truth_keys:
        print('FITTING %s TRANSFORMER' %key)
        scaler          = RobustScaler()
        scaler          = scaler.fit(np.array(truth[key]).reshape(-1,1))
        truth[key]      = scaler.transform(np.array(truth[key]).reshape(-1,1))
        transformer_dict['truth'][key] = scaler
    
    ####################################
    #                                  #
    #      CREATE SQLite DATABASE      #
    #                                  #
    ####################################
    print('CREATING DATABASE')
    save = True
    
    transformer_path = outdir + '\\' + '\\%s\\'%db_name + 'meta'
    db_path          = outdir + '\\' + '\\%s\\'%db_name + 'data'
 
    os.makedirs(db_path, exist_ok= False )
  
    os.makedirs(transformer_path, exist_ok= False )

                
    if save == True:
        print('SAVING DATABASE..')
        conn = sqlite3.connect(db_path + '\\%s.db'%db_name)
        c = conn.cursor()
        truth.to_sql('truth',conn,index= False)
        features.to_sql('input',conn,index= False)
        print('SAVING TRANSFORMERS..')
        with open(transformer_path + '\\transformers.pkl', 'wb') as handle:
            pickle.dump(transformer_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('DONE!')    
    return

if __name__ == '__main__':
    CreateDataBase(**vars(parse_args()))

