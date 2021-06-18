import sqlite3
import numpy as np
import pandas as pd

db_file = r'J:\speciale\data\raw\dev_numu_train_upgrade_step4_2020_00\data\dev_numu_train_upgrade_step4_2020_00.db'



feats = str('event_no,dom_x,dom_y,dom_z,dom_time,dom_charge,dom_string,dom_pmt,dom_om,dom_lc,dom_atwd,dom_fadc')
sca = pd.DataFrame()                                                             #
seq = pd.DataFrame()
with sqlite3.connect(db_file) as con:                                           #
    query = 'select %s from features limit 100'%feats                                        # MERGES ALL .db FILES TO TWO .csv FILES:
    seq = seq.append(pd.read_sql(query, con))                       
    query = 'select * from truth limit 1 '                                            # THESE ARE THEN WRITTEN TO DRIVE
    sca = sca.append(pd.read_sql(query, con))                             #
    #cursor = con.cursor()                                                       #
    #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")