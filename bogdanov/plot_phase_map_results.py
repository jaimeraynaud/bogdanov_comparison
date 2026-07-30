import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

# Get the project root directory (parent of bogdanov)
PROJECT_ROOT = Path(__file__).parent.parent
COMPARE_DIR = PROJECT_ROOT / 'bogdanov'
IMG_DIR = PROJECT_ROOT / 'images/bogdanov'

def dat_to_csv(dat_path, csv_path=None, delimiter=None, header=False):
    '''
    Convert a .dat text table to CSV and return the output CSV path.

    :param dat_path: input .dat file path
    :param csv_path: output .csv file path (defaults to input stem + .csv)
    :param delimiter: explicit delimiter; if None, use whitespace splitting
    :param header: whether to write column names in output CSV
    :return: output csv path as string
    '''
    dat_path = Path(dat_path)
    if csv_path is None:
        csv_path = dat_path.with_suffix('.csv')
    else:
        csv_path = Path(csv_path)

    # Most .dat files in this workflow are whitespace-delimited.
    if delimiter is None:
        df = pd.read_csv(dat_path, sep=r'\s+', engine='python', header=None)
    else:
        df = pd.read_csv(dat_path, sep=delimiter, engine='python', header=None)

    df.to_csv(csv_path, index=False, header=header)
    return str(csv_path)

expected = pd.read_csv(COMPARE_DIR / 'two_spot_synthetic_expected.dat', sep=' ',header=None)
realisation = pd.read_csv(COMPARE_DIR / 'two_spot_synthetic_realisation.dat', sep=' ',header=None)

spot1_wendy = pd.read_csv(COMPARE_DIR/'phasemap_spot1_counts_v2.csv')
spot2_wendy = pd.read_csv(COMPARE_DIR/'phasemap_spot2_counts_v2.csv')


# spot1 = pd.read_csv(dat_to_csv(COMPARE_DIR / 'reproducing/spot1_test_data_counts.dat', COMPARE_DIR / 'responsefunc/spot1_test_data_counts.csv'), header=None)
# spot2 = pd.read_csv(dat_to_csv(COMPARE_DIR / 'reproducing/spot2_test_data_counts.dat', COMPARE_DIR / 'responsefunc/spot2_test_data_counts.csv'), header=None)

# spot1 = pd.read_csv(dat_to_csv(COMPARE_DIR / 'latest/spot1_test_data_counts.dat', COMPARE_DIR / 'latest/spot1_test_data_counts.csv'), header=None)
# spot2 = pd.read_csv(dat_to_csv(COMPARE_DIR / 'latest/spot2_test_data_counts.dat', COMPARE_DIR / 'latest/spot2_test_data_counts.csv'), header=None)

spot1 = pd.read_csv(dat_to_csv(COMPARE_DIR / 'reproducing/spot1_test_data_counts.dat', COMPARE_DIR / 'reproducing/spot1_test_data_counts.csv'), header=None)
spot2 = pd.read_csv(dat_to_csv(COMPARE_DIR / 'reproducing/spot2_test_data_counts.dat', COMPARE_DIR / 'reproducing/spot2_test_data_counts.csv'), header=None)

spot1_new = pd.read_csv(dat_to_csv(COMPARE_DIR / 'final/spot1_test_data_counts.dat', COMPARE_DIR / 'final/spot1_test_data_counts.csv'), header=None)
spot2_new = pd.read_csv(dat_to_csv(COMPARE_DIR / 'final/spot2_test_data_counts.dat', COMPARE_DIR / 'final/spot2_test_data_counts.csv'), header=None)

both_spots_wendy = spot1_wendy + spot2_wendy
both_spots = spot1_new + spot2_new

background = pd.read_csv(COMPARE_DIR / 'background_testcase2_1e6counts_25_299.dat', sep=' ',header=None)
# absorption= pd.read_csv('/Users/wfwallac/Downloads/two_spot_synthetic_NICER_data/tbabs/tbnew0.02.txt', sep=' ',header=None)

#add background in for comparison to the model data
for i in range(32):
#     both_spots[i,:] = both_spots[i,:] + background[0]/32
    both_spots.iloc[i] = np.sum((both_spots.iloc[i], background[0] / 32), axis=0)
    both_spots_wendy.iloc[i] = np.sum((both_spots_wendy.iloc[i], background[0] / 32), axis=0)
#     phase_map.iloc[i] = np.sum((phase_map.iloc[i], background_counts / 32), axis=0)

#shift and define bins for energy-phase maps
phase_bins = np.linspace(0.0, 1-1/32, 32)
energy_bins = np.linspace(26, 300, 275)
test_data=np.array(realisation[2]).reshape(275,32)
both_spots=np.array(both_spots)
both_spots_wendy=np.array(both_spots_wendy)
shifted_phase= np.concat((both_spots[8:,:275],both_spots[:8,:275]),axis=0) #18 for 300rs, [7/8,:700] for 600rs
total_photons = shifted_phase.sum(axis=0).sum(axis=0)
shifted_phase_wendy = np.concat((both_spots_wendy[8:,:275],both_spots_wendy[:8,:275]),axis=0) #18 for 300rs, [7/8,:700] for 600rs
rel_diff=np.abs((test_data-shifted_phase.T))/test_data
normalized_absolute_diff = np.abs((test_data-shifted_phase.T))/total_photons
rel_diff_wendy=(test_data-shifted_phase_wendy.T)/test_data

### Plot your results
# plot = plt.pcolormesh(phase_bins, energy_bins, shifted_phase.T, cmap='magma', shading='nearest')
# plt.xlabel('phase')
# plt.ylabel('energy channel')
# plt.title('Jaimes Phase Map (Using fortran code)')
# plt.colorbar(plot)
# # plt.plot(phase_bins, demo.sum(axis=1))
# plt.savefig(IMG_DIR / 'phase_map_jaime.png')
# plt.show()

# plot = plt.pcolormesh(phase_bins, energy_bins, shifted_phase_wendy.T, cmap='magma', shading='nearest')
# plt.xlabel('phase')
# plt.ylabel('energy channel')
# plt.title('Wendys Phase Map')
# plt.colorbar(plot)
# # plt.plot(phase_bins, demo.sum(axis=1))
# plt.savefig(IMG_DIR / 'phase_map_wendy.png')
# plt.show()

# ## Plot model data
plot=plt.pcolormesh(phase_bins,energy_bins,test_data,cmap='magma',shading='nearest',norm=LogNorm())
plt.xlabel('phase')
plt.ylabel('energy channel')
plt.title('Bogdanov Phase Map')
plt.colorbar(plot)
plt.savefig(IMG_DIR / 'phase_map_bogdanov.png')
plt.show()

# ### Plot relative difference between the two data outputs
# plot=plt.pcolormesh(phase_bins,energy_bins,rel_diff,cmap='coolwarm',shading='nearest')
# plt.title('Relative Difference Phase Map')
# plt.xlabel('phase')
# plt.ylabel('energy channel')
# plt.colorbar(plot)
# plt.show()
#
# ### Plot absolute difference between the two data outputs
# plot=plt.pcolormesh(phase_bins,energy_bins,normalized_absolute_diff,cmap='coolwarm',shading='nearest')
# plt.title('Normalized Absolute Difference Phase Map')
# plt.xlabel('phase')
# plt.ylabel('energy channel')
# plt.colorbar(plot)
# plt.show()

# ### Scatter plot of relative difference vs test_data (flattened)
# plt.figure(figsize=(10, 6))
# plt.scatter(test_data.flatten(), rel_diff.flatten(), alpha=0.5, s=10)
# plt.xlabel('Test Data (Observed Counts)')
# plt.ylabel('Relative Difference')
# plt.title('Relative Difference vs Test Data')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

phase_bins_ck=(phase_bins+2/64.)%1.0
print(phase_bins)
print(phase_bins_ck)
# new_order = np.argsort(np.arange(len(x)) % modulus)
# # 3. Apply the same reordering to both
# x_new = x[new_order]
# y_new = y[new_order]
### Plot bolometric lightcurve (or switch sum axes and change to energy_bins for comparison of E distrib)
summed_vals=shifted_phase.sum(axis=1)
summed_vals2=test_data.sum(axis=0)
plt.plot(phase_bins,summed_vals,label='Non BF output')
summed_vals_wendy=shifted_phase_wendy.sum(axis=1)
plt.plot(phase_bins,summed_vals_wendy,label='Brute Force output')
# plt.plot(phase_bins,summed_vals2,label='Bogdanovs testcase')
plt.title('Bogdanovs Bolometric LC')
plt.legend()
plt.savefig(IMG_DIR / 'bogdanov_bolometric.png')
plt.show()

### Plot bolometric lightcurve (or switch sum axes and change to energy_bins for comparison of E distrib)
# summed_vals_wendy=shifted_phase_wendy.sum(axis=1)
# summed_vals2=test_data.sum(axis=0)
# plt.plot(phase_bins,summed_vals_wendy,label='Wendy method')
# plt.plot(phase_bins,summed_vals2,label='expected testcase')
# plt.title('Wendys Bolometric LC')
# plt.legend()
# plt.savefig(IMG_DIR / 'bolometric_wendy.png')
# plt.show()

pass