import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


results = r'X:\speciale\results\dev_numu_train_l5_retro_001\event_only_aggr_test_SRT_energy_scaled_new_weights\dynedgev3-azimuth-cosine-pair-tanh\results.csv'

scalers = pd.read_pickle(r'X:\speciale\data\raw\dev_numu_train_l5_retro_001\meta\transformers.pkl')



results = pd.read_csv(results)

results['dir_x'] = scalers['truth']['direction_x'].inverse_transform(np.array(results['dir_x']).reshape(-1,1))
results['dir_y'] = scalers['truth']['direction_y'].inverse_transform(np.array(results['dir_y']).reshape(-1,1))
results['dir_z'] = scalers['truth']['direction_z'].inverse_transform(np.array(results['dir_z']).reshape(-1,1))

#results['dir_x_pred'] = scalers['truth']['direction_x'].inverse_transform(np.array(results['dir_x_pred']).reshape(-1,1))
#results['dir_y_pred'] = scalers['truth']['direction_y'].inverse_transform(np.array(results['dir_y_pred']).reshape(-1,1))
#results['dir_z_pred'] = scalers['truth']['direction_z'].inverse_transform(np.array(results['dir_z_pred']).reshape(-1,1))


results['azimuth'] = scalers['truth']['azimuth'].inverse_transform(np.array(results['azimuth']).reshape(-1,1))
results['dir_azimuth_pred'] = scalers['truth']['azimuth'].inverse_transform(np.array(results['dir_azimuth_pred']).reshape(-1,1))


#dot_prod = (results['dir_x']*results['dir_x_pred'] + results['dir_y']*results['dir_y_pred'] + results['dir_z']*results['dir_z_pred'])

#norms = (results['dir_x']**2 + results['dir_y']**2 + results['dir_z']**2)*(results['dir_x_pred']**2 + results['dir_y_pred']**2 + results['dir_z_pred']**2)



#angle_between = np.arccos(dot_prod/norms)*(360/(2*np.pi))

norm_true = (results['dir_x']**2 + results['dir_y']**2 + results['dir_z']**2)

n = 100000

zenith = np.arccos(-results['dir_z'])[0:n]

#plt.scatter(results['zenith'][0:n],zenith )

#plt.scatter(results['zenith'][0:n],np.arccos(-results['dir_z_pred'])[0:n] )

plt.hist2d(results['azimuth'],results['azimuth_pred'],bins=100)