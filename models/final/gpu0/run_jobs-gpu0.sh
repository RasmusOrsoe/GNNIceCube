#python dynedge_zenith_no_scale.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_e'
#python dynedge_zenith.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_e'
#python dynedge_zenith_no_scale.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_tau'
#python dynedge_zenith.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_tau'
#python dynedge_zenith_no_scale.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_muon'
#python dynedge_zenith.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_muon'

#python dynedge_azimuth_no_scale.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_e'
#python dynedge_azimuth.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_e'
#python dynedge_azimuth_no_scale.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_tau'
#python dynedge_azimuth.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_tau'
#python dynedge_azimuth_no_scale.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_muon'
#python dynedge_azimuth.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_muon'

#python dynedge_azimuth_no_scale.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_1mio'
#python dynedge_azimuth.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_1mio'

#python dynedge_zenith_no_scale.py --k 12 --graphs 'dev_level7_mu_e_tau_oscweight_000/event_only_level7_all_oscweight_1mio'

#python ~/speciale/models/dynedge_ensemble/likelihood/dynedge_likelihood_azimuth_protov3.py 
#python dynedge_zenith.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/all_e'
#python dynedge_energy.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/all_e'
#python dynedge_energy.py --k 8 --tag 'thesis-oldlr-50e-check' --graphs 'oscnext_IC8611_newfeats_000_mc_scaler/regression'
#python dynedge_azimuth.py --k 8 --tag 'thesis-newlr-30e-check' --graphs 'oscnext_IC8611_newfeats_000_mc_scaler/regression'
#python dynedge_zenith.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/1mio_mu_only_zenith_even'
#python dynedge_energy.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/all_tau'
#python dynedge_azimuth.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/all_tau'

#python dynedge_zenith.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/mix_2mio'
#python dynedge_energy.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/mix_2mio'
#python dynedge_azimuth.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/mix_2mio'

#python dynedge_zenith.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/2mio_muons_only'
#python dynedge_energy.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/2mio_muons_only'
#python dynedge_azimuth.py --k 8 --tag 'thesis-newlr-30e' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/2mio_muons_only'

#python dynedge_classification.py --k 8 --device 'cuda:1' --tag 'classification' --graphs 'dev_level2_mu_tau_e_muongun_classification_wnoise/3mio_even_noise_classifier'
#python dynedge_classification.py --k 8 --device 'cuda:1' --tag 'classification' --graphs 'oscnext_IC8611_newfeats_000_mc_scaler/burnsample'

#python dynedge_energy.py --k 8 --tag 'w_test_val' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/everything_wtest_testset' --device 'cuda:1'
python dynedge_classification.py --k 8 --tag 'track_cascade_good' --graphs 'dev_level7_mu_e_tau_oscweight_newfeats/track_cascadev2' --device 'cuda:0'

#python dynedge_azimuth.py --k 8 --tag 'thesis-newlr-30e' --graphs 'IC8611_oscNext_003_final/whole_sample' --device 'cuda:0'
pause > nul
