import sqlite3
import pandas as pd
from collections import Counter
db_file = r'J:\speciale\data\raw\standard\dev_numu_train_l2_2020_01.db'                    #
sca = pd.DataFrame()                                                             #
seq = pd.DataFrame()
events = pd.read_csv(r'J:\speciale\data\raw\standard\sliced\even_events.csv')
events_val = pd.read_csv(r'J:\speciale\data\raw\standard\sliced\even_events_val.csv')
events = events.append(events_val).loc[:,'event_no'].values
feats = str('event_no,dom_x,dom_y,dom_z,dom_time,dom_charge')
#truths = str('energy_log10,time,position_x,position_y,position_z,direction_x,direction_y,direction_z,azimuth,zenith')                                                      #                                                      #
with sqlite3.connect(db_file) as con:                                           #
    query = 'select %s from features WHERE event_no IN %s'%(feats,str(tuple(events)))                                        # MERGES ALL .db FILES TO TWO .csv FILES:
    seq = seq.append(pd.read_sql(query, con))                     # scalar.csv , sequential.csv   
seq = seq.drop('event_no')
