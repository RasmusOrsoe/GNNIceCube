'''
Plot results of `test_reco_hypo.py`

Tom Stuttard
'''

import os, sys, collections, glob

from utils.plotting.standard_modules import *

from icecube import dataclasses, icetray, dataio

def extract_fit_vector(frame, key, omlist, numoms) :

    charges = collections.OrderedDict()
    
    times = collections.OrderedDict()



    data = frame[key]

    
    for om_key in list(omlist.keys())[:numoms] :
    
        charges[om_key] = np.array( [ p.charge for p in data[om_key] ] )
        times[om_key] = np.array( [ p.time for p in data[om_key] ] )
        
    return charges, times


#
# Main
#

if __name__ == "__main__" :

    from utils.script_tools import ScriptWrapper
    from utils.filesys_tools import replace_file_ext, get_file_stem
    with ScriptWrapper(log_file=replace_file_ext(__file__,".log")) as script :
        
        
        
        #
        # Get inputs
        #
        
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-os", "--oversampling", type=int, required=True)
        args = parser.parse_args()
        # Events
        input_files = sorted(glob.glob("/groups/icecube/kpederse/test_reco_hypo/test_mc_PE_only/test_recohypo_stability_2_os_"+str(args.oversampling)+"/*.i3.zst"))
        i3_file = dataio.I3File(input_files[0], "r")
        frame = i3_file.pop_physics()
        #
        # Load file
        #
        #print(dir(frame["I3EventHeader"]) )
        event_id = frame["I3EventHeader"].sub_run_id
        print(event_id)
        # Geom
        GCD_FILE = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
        gcd_file = dataio.I3File(GCD_FILE)
        g_frame = gcd_file.pop_frame(icetray.I3Frame.Geometry)
        om_geom_dict = g_frame["I3Geometry"].omgeo
        
        #Get oms for plotting
        # Get vertex position
        
        vertex = frame["I3MCTree"][0].pos

        # Get a list of vertex-OM distances
        om_vertex_distances = []
        for om_key, om_geom in om_geom_dict.items() :
            om_vertex_distances.append( (vertex - om_geom.position).magnitude )

        # Sort and put into a dict for use later
        sort_indices = np.argsort(om_vertex_distances)
        closest_oms = collections.OrderedDict()
        for idx in sort_indices :
            closest_oms[ list(om_geom_dict.keys())[idx] ] = om_vertex_distances[idx]


        # Steering
        stem = "MillipedeHypothesis_DirectReco_"
        cases = [ "Data", "Hypo" ]
        num_events = 100
        num_oms_to_plot = 25
        
        
        nx, ny = get_grid_dims(n=num_oms_to_plot)
        fig = Figure( nx=ny, ny=ny )
        binning = get_bins(9800.,10500.,width = 8.)
        # Useful bits
        event_counter = 0
        
        from matplotlib.pyplot import cm
        col=iter(cm.rainbow(np.linspace(0,1,num_events)))
        
        for input_file in input_files:
        
            i3_file = dataio.I3File(input_file, "r")
            # Start loop...
            while i3_file.more() :

                frame = i3_file.pop_physics()

                if frame :
                    event_counter += 1
                    if event_counter > num_events :
                        break

                    print(frame["I3EventHeader"])
                    
                    data_charges,data_times = extract_fit_vector(frame, stem + cases[0],closest_oms, num_oms_to_plot)
                    hypo_charges,hypo_times = extract_fit_vector(frame, stem + cases[1],closest_oms, num_oms_to_plot)


                    #
                    # Plot
                    #



                    # Loop over the closest OMs
                    om_keys = list(closest_oms.keys())[:num_oms_to_plot]
                    c = next(col)
                    print(c)
                    for i_om, om_key in enumerate(om_keys) :

                        ax = fig.get_ax(i=i_om)
                        ax.set_title("String %i OM %i (%0.3g m from vertex)" % (om_key.string, om_key.om, closest_oms[om_key]) )

                        # Plot hypo and data
                        data_hist = Histogram(ndim = 1, uof = False, bins = binning, x = data_times[om_key], weights = data_charges[om_key])
                        hypo_hist = Histogram(ndim = 1, uof = False, bins = binning, x = hypo_times[om_key], weights = hypo_charges[om_key])
                        
                        plot_hist(ax=ax, hist=data_hist, color="black",alpha = 1., linestyle="-", errors=False, label="Data")
                        plot_hist(ax=ax, hist=hypo_hist, color=c,alpha = 0.2, linestyle="--", errors=False, label="Hypo")
        fig.quick_format(xlabel="t [ns]", ylabel="Charge [p.e.]", ylim=(0., None),legend=False, rect=[0.,0.,1.,0.95])
        fig.fig.suptitle("DirectReco multiple hypothesees for event {}, Oversampling {}\n Data = Black, Hypo = Colors".format(event_id, args.oversampling))
                    



        #
        # Done
        #

        print("")
        dump_figures_to_pdf( replace_file_ext(__file__,".pdf") )

