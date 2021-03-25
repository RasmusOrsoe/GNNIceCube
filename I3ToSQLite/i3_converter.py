'''

'''

import os, sys, collections, glob

import numpy as np
import pandas as pd

#scalers  = pd.read_pickle('/groups/hep/pcs557/i3_workspace/scalers/dev_classification_000/meta/transformers.pkl')

from sqlalchemy import create_engine
import sqlalchemy
from icecube import dataclasses, icetray, dataio
import time
import pickle
from multiprocessing import Pool
import multiprocessing

def extract_fit_vector(frame, key, gcd_dict,mode,calibration) :


    charge = []
    
    time   = []

    width  = []
    area   = []
    rqe    = []

    pos     = []
    x       = []
    y       = []
    z       = []

    #print(frame.keys())

    #frame['fail']
   
    if 'I3MCTree' in frame.keys() and key in frame.keys() and 'mc' in mode:

    
        truths = frame['I3MCTree'][0]
        data    = frame[key]
        #print(data)
        
        event_time =  frame['I3EventHeader'].start_time.utc_daq_time
        interaction_type =  frame["I3MCWeightDict"]["InteractionType"]
        
        
        #try:
        #    om_keys = data.keys()
        #except:
        #    try:
        #        data = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,key)
        #        om_keys = data.keys()
        #    except:
        #        print('mask fail..')
        #        return
        try:
            om_keys = data.keys()
        except:
            try:
                frame["I3Calibration"] = calibration
                
                print('manually inserted calibration')
                data = frame[key].apply(frame)
                om_keys = data.keys()
            except:
                data = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,key)
                om_keys = data.keys()
        
        #print(dir(frame["I3Calibration"]))
        for om_key in om_keys:
            pulses = data[om_key]
            for pulse in pulses:
                charge.append(pulse.charge)
                time.append(pulse.time) 
                width.append(pulse.width)
                area.append(gcd_dict[om_key].area)  
                rqe.append(frame["I3Calibration"].dom_cal[om_key].relative_dom_eff)
                x.append(gcd_dict[om_key].position.x)
                y.append(gcd_dict[om_key].position.y)
                z.append(gcd_dict[om_key].position.z)
                
        if mode == 'mc':
            truth = [truths.energy, truths.pos.x,truths.pos.y
                , truths.pos.z, truths.dir.azimuth,
                truths.dir.zenith, truths.pdg_encoding,event_time, interaction_type]

        if mode == 'mc-retro':
            weight = frame['I3MCWeightDict']["weight"]
            interaction_type =  frame["I3MCWeightDict"]["InteractionType"] 
            truth = [truths.energy, 
                        truths.pos.x,
                        truths.pos.y, 
                        truths.pos.z, 
                        truths.dir.azimuth,
                        truths.dir.zenith, 
                        truths.pdg_encoding,
                        event_time,
                        frame['L7_reconstructed_azimuth'],
                        frame['L7_reconstructed_time'],
                        frame['L7_reconstructed_total_energy'],
                        frame['L7_reconstructed_vertex_x'],
                        frame['L7_reconstructed_vertex_y'],
                        frame['L7_reconstructed_vertex_z'],
                        frame['L7_reconstructed_zenith'],
                        frame['L7_retro_crs_prefit__azimuth_sigma_tot'],
                        frame['L7_retro_crs_prefit__x_sigma_tot'],
                        frame['L7_retro_crs_prefit__y_sigma_tot'],
                        frame['L7_retro_crs_prefit__z_sigma_tot'],
                        frame['L7_retro_crs_prefit__time_sigma_tot'],
                        frame['L7_retro_crs_prefit__zenith_sigma_tot'],
                        frame['L7_retro_crs_prefit__energy_sigma_tot'],
                        weight,
                        interaction_type]
        
            
            
        return charge, time, width,area, rqe, x, y ,z , truth
    

    if mode == 'data':
        print('extracting data')

        truth = [frame['L7_reconstructed_azimuth'],
                 frame['L7_reconstructed_time'],
                 frame['L7_reconstructed_total_energy'],
                 frame['L7_reconstructed_vertex_x'],
                 frame['L7_reconstructed_vertex_y'],
                 frame['L7_reconstructed_vertex_z'],
                 frame['L7_reconstructed_zenith'],
                 frame['L7_retro_crs_prefit__azimuth_sigma_tot'],
                 frame['L7_retro_crs_prefit__x_sigma_tot'],
                 frame['L7_retro_crs_prefit__y_sigma_tot'],
                 frame['L7_retro_crs_prefit__z_sigma_tot'],
                 frame['L7_retro_crs_prefit__time_sigma_tot'],
                 frame['L7_retro_crs_prefit__zenith_sigma_tot'],
                 frame['L7_retro_crs_prefit__energy_sigma_tot']]
    
        data    = frame[key]
        #print(data)
        
        event_time =  frame['I3EventHeader'].start_time.utc_daq_time
        
        try:
            om_keys = data.keys()
        except:
            try:
                frame["I3Calibration"] = calibration
                print('manually inserted calibration')
                data = frame[key].apply(frame)
                om_keys = data.keys()
            except:
                data = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,key)
                om_keys = data.keys()
                
     
        for om_key in om_keys:
            pulses = data[om_key]
            for pulse in pulses:
                charge.append(pulse.charge)
                time.append(pulse.time) 
                width.append(pulse.width)
                area.append(gcd_dict[om_key].area)  
                rqe.append(frame["I3Calibration"].dom_cal[om_key].relative_dom_eff)
                x.append(gcd_dict[om_key].position.x)
                y.append(gcd_dict[om_key].position.y)
                z.append(gcd_dict[om_key].position.z)
                
        
    
        return charge, time, width,area,rqe, x, y ,z,truth
    
    else:
        #print('I3MCTree or KEY NOT FOUND. SKIPPIN...')
        return None

def WriteDicts(settings):
    
    input_files,id,gcd_files,outdir , max_dict_size,event_no_list,mode = settings
    # Useful bits
    event_counter = 0
    output = []
    feature_big = pd.DataFrame()
    truth_big   = pd.DataFrame()
    file_counter = 0
    output_count = 0
    gcd_count = 0
    print(len(gcd_files))
    print(len(input_files))
    print(len(event_no_list))
    #print(input_files)
    for u  in range(0,len(input_files)):
        input_file = input_files[u]
        print('loading gcd')
        gcd_file = dataio.I3File(gcd_files[u])
        g_frame = gcd_file.pop_frame(icetray.I3Frame.Geometry)
        om_geom_dict = g_frame["I3Geometry"].omgeo
        calibration = gcd_file.pop_frame(icetray.I3Frame.Calibration)["I3Calibration"]    
        i3_file = dataio.I3File(input_file, "r")
        
        print('gcd loaded')

        print('Using %s for '%gcd_files[u].split('/')[-1])
        print('for i3-file %s'%input_file.split('/')[-1])
        gcd_count  +=1
        #print(dir(i3_file))
        # Start loop...
        sim_type = 'lol'
        if 'muon' in input_file:
            sim_type = 'muongun'
        if 'corsika' in input_file:
            sim_type = 'corsika'
        if 'genie' in input_file:
            sim_type = 'genie'
        if 'real' in input_file:
            sim_type = 'data'
        if sim_type == 'lol':
            print('SIM TYPE NOT FOUND!')
        while i3_file.more() :

            try:
                frame = i3_file.pop_physics()
            except:
                frame = False

            if frame :
                
                #print((frame.keys()))
                
                out = extract_fit_vector(frame,'SRTInIcePulses',om_geom_dict,mode,calibration)
                

                if out != None and 'mc' in mode:
                    charge,time, width,area, rqe, x ,  y ,  z , truth  = out
                    features = {'charge_log10': charge, 'dom_time': time, 
                                'dom_x': x, 'dom_y': y, 'dom_z': z,
                                'width' : width, 'pmt_area': area, 'rqe': rqe}
                    if mode == 'mc':
                        truths = {'energy_log10': truth[0],
                                    'position_x': truth[1], 
                                    'position_y': truth[2], 
                                    'position_z': truth[3],
                                    'azimuth': truth[4],
                                    'zenith': truth[5],
                                    'pid': truth[6],
                                    'event_time': truth[7],
                                    'sim_type': sim_type,
                                    'interaction_type': truth[8]}
                    if mode == 'mc-retro':
                        truths = {'energy_log10': np.log10(truth[0]),
                                    'position_x': truth[1], 
                                    'position_y': truth[2], 
                                    'position_z': truth[3],
                                    'azimuth': truth[4],
                                    'zenith': truth[5],
                                    'pid': truth[6],
                                    'event_time': truth[7],
                                    'sim_type': sim_type,
                                    'azimuth_retro': truth[8].value,
                                    'time_retro': np.log10(truth[9].value),
                                    'energy_log10_retro': np.log10(truth[10].value), 
                                    'position_x_retro': truth[11].value, 
                                    'position_y_retro': truth[12].value,
                                    'position_z_retro': truth[13].value,
                                    'zenith_retro': truth[14].value,
                                    'azimuth_sigma': truth[15].value,
                                    'position_x_sigma': truth[16].value,
                                    'position_y_sigma': truth[17].value,
                                    'position_z_sigma': truth[18].value,
                                    'time_sigma': truth[19].value,
                                    'zenith_sigma': truth[20].value,
                                    'energy_log10_sigma': truth[21].value,
                                    'osc_weight': truth[22],
                                    'interaction_type':truth[23]}

                            
                    truth = pd.DataFrame(truths.values()).T
                    truth.columns = truths.keys()
                    truth['event_no'] = event_no_list[event_counter]

                    feature = pd.DataFrame(features.values()).T
                    feature.columns = features.keys()
                    feature['event_no'] = event_no_list[event_counter]

                    event_counter += 1


                    feature_big = feature_big.append(feature,ignore_index = True)
                    truth_big   = truth_big.append(truth, ignore_index = True)

                    if len(truth_big) >= max_dic_size:
                        print('savin')

                        engine = sqlalchemy.create_engine('sqlite:///'+outdir + '/worker-%s-%s.db'%(id,output_count))
                        truth_big.to_sql('truth',engine,index= False, if_exists = 'append')
                        print('truth in')
                        feature_big.to_sql('features',engine,index= False, if_exists = 'append')
                        engine.dispose()
                        print('features in')

                        feature_big = pd.DataFrame()
                        truth_big   = pd.DataFrame()
                        print('saved')
                        output_count +=1

                if out != None and mode == 'data':
                    
                    charge,time, width,area, rqe, x ,  y ,  z , retro  = out
                    features = {'charge_log10': charge, 'dom_time': time, 
                                'dom_x': x, 'dom_y': y, 'dom_z': z,
                                'width' : width, 'pmt_area': area, 'rqe': rqe}
                    print(retro[1].value)
                    truths = {'azimuth_retro': retro[0].value,
                                'time_retro': np.log10(retro[1].value),
                                'energy_log10_retro': np.log10(retro[2].value), 
                                'position_x_retro': retro[3].value, 
                                'position_y_retro': retro[4].value,
                                'position_z_retro': retro[5].value,
                                'zenith_retro': retro[6].value,
                                'azimuth_sigma': retro[7].value,
                                'position_x_sigma': retro[8].value,
                                'position_y_sigma': retro[9].value,
                                'position_z_sigma': retro[10].value,
                                'time_sigma': retro[11].value,
                                'zenith_sigma': retro[12].value,
                                'energy_log10_sigma': retro[13].value}

                    feature = pd.DataFrame(features.values()).T
                    feature.columns = features.keys()
                    feature['event_no'] = event_no_list[event_counter]

                    truth = pd.DataFrame(truths.values()).T
                    truth.columns = truths.keys()
                    truth['event_no'] = event_no_list[event_counter]
                    event_counter += 1


                    feature_big = feature_big.append(feature,ignore_index = True)
                    truth_big   = truth_big.append(truth, ignore_index = True)

                    if  len(truth_big) >= max_dic_size:
                        print('savin')

                        engine = sqlalchemy.create_engine('sqlite:///'+outdir + '/worker-%s-%s.db'%(id,output_count))
                        truth_big.to_sql('truth',engine,index= False, if_exists = 'append')
                        print('retro in')

                        engine = sqlalchemy.create_engine('sqlite:///'+outdir + '/worker-%s-%s.db'%(id,output_count))
                        feature_big.to_sql('features',engine,index= False, if_exists = 'append')
                        engine.dispose()
                        print('features in')

                        feature_big = pd.DataFrame()
                        truth_big   = pd.DataFrame()
                        print('saved')
                        output_count +=1
        print('WORKER %s : %s/%s'%(id,file_counter,len(input_files)))
        file_counter +=1
    print(len(feature_big))
    engine = sqlalchemy.create_engine('sqlite:///'+outdir + '/worker-%s-%s.db'%(id,output_count))

    truth_big.to_sql('truth',engine,index= False, if_exists = 'append')
    feature_big.to_sql('features',engine,index= False, if_exists = 'append')
    engine.dispose()
    feature_big = pd.DataFrame()
    truth_big   = pd.DataFrame()
#
# Main
#

if __name__ == "__main__" :
  
    start_time = time.time()    
    #
    # Get inputs
    #
    
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("-path", "--path", type=str, required=True)
    parser.add_argument("-outdir", "--outdir", type=str, required=True)
    parser.add_argument("-workers", "--workers", type=int, required=True)
    parser.add_argument("-mode", "--mode", type=str, required=True)
    args = parser.parse_args()
    #paths = ['/groups/hep/pcs557/i3_workspace/data/corsika_2020/corsika_2020/00000-00999',
    #        '/groups/hep/pcs557/i3_workspace/data/oscNext/genie/level2/120000',
    #        '/groups/hep/pcs557/i3_workspace/data/oscNext/genie/level2/140000',
    #        '/groups/hep/pcs557/i3_workspace/data/oscNext/genie/level2/160000']
            #'/groups/hep/pcs557/i3_workspace/data/oscNext/muongun/level2/139008']
    
    path =  '/groups/hep/pcs557/i3_workspace/data/real_data/level7_v02.00/IC86.11' #'/groups/hep/pcs557/i3_workspace/data/oscNext/genie/level7_v02.00' # 

    extensions = ("/*.i3.bz2","/*.zst","/*.gz")
    input_files_mid = []
    input_files = []
    files = []
    gcd_files_mid = []
    gcd_files = []
    paths =  glob.glob(path + '/*')
    gcd_rescue = '/groups/icecube/stuttard/data/oscNext/pass2/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
    print(paths)
    
    for path in paths:
        for extension in extensions:
            mid = glob.glob(path + extension)
            if mid != []:
                for file in mid:
                    if 'gcd' in file:
                        gcd_file = file

                    if 'GCD' in file:
                        gcd_file = file
                    
                    if 'geo' in file:
                        gcd_file = file
                    if 'Geo' in file:
                        gcd_file = file
                    else:
                        input_files_mid.append(file)
        try:
             print(gcd_file)
        except:
            gcd_file = gcd_rescue

        for k in range(0,len(input_files_mid)):
            gcd_files_mid.append(gcd_file)
        input_files.extend(input_files_mid)
        gcd_files.extend(gcd_files_mid)
        gcd_files_mid = []
        input_files_mid = []
        
        


    print('gcd files: %s'%len(gcd_files))
    print('i3 files: %s'%len(input_files))


    outdir = args.outdir
    try:
        os.makedirs(outdir)
    except:
        print(' !!WARNING!! \n \
            %s already exists. \n \
        Abort to avoid overwriting! '%args.outdir)

    # SETTINGS
    max_dic_size  =  10000
    #GCD_FILE =  '/groups/hep/pcs557/i3_workspace/data/real_data/level7_v02.00/IC86.11/Run00118909/Level2pass2_IC86.2011_data_Run00118909_1112_1_20_GCD.i3.zst'  ##'/groups/icecube/stuttard/data/oscNext/pass2/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'#"/groups/hep/pcs557/i3_workspace/data/corsika_2012/00000-00999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz" # 
    

    settings = []
    event_nos = np.array_split(np.arange(0,120102888,1),args.workers)
    file_list = np.array_split(np.array(input_files),args.workers)
    gcd_file_list = np.array_split(np.array(gcd_files),args.workers)
    for i in range(0,args.workers):
        settings.append([file_list[i],str(i),gcd_file_list[i],args.outdir,max_dic_size,event_nos[i],args.mode])
    #WriteDicts(settings[0])
    p = Pool(processes = args.workers)
    p.map(WriteDicts, settings)   
    p.close()
    p.join()
    
    print('Job Complete! Time Elapsed: %s min' %((time.time() - start_time)/60))


    

    
    
                    



             






