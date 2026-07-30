import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from astropy.constants.codata2022 import alpha
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

# Get the project root directory (parent of bogdanov)
PROJECT_ROOT = Path(__file__).parent.parent
COMPARE_DIR = PROJECT_ROOT / 'bogdanov'
IMG_DIR = PROJECT_ROOT / 'images'

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

spot1 = pd.read_csv(dat_to_csv(COMPARE_DIR / 'combined/spot1_test_data_counts.dat', COMPARE_DIR / 'combined/spot1_test_data_counts.csv'), header=None)
spot2 = pd.read_csv(dat_to_csv(COMPARE_DIR / 'combined/spot2_test_data_counts.dat', COMPARE_DIR / 'combined/spot2_test_data_counts.csv'), header=None)
summed_spots = spot1 + spot2

combined_spots = pd.read_csv(dat_to_csv(COMPARE_DIR / 'combined/combined_test_data_counts.dat', COMPARE_DIR / 'combined/combined_test_data_counts.csv'), header=None)


background = pd.read_csv(COMPARE_DIR / 'background_testcase2_1e6counts_25_299.dat', sep=' ',header=None)

#add background in for comparison to the model data
for i in range(32):
#     both_spots[i,:] = both_spots[i,:] + background[0]/32
    summed_spots.iloc[i] = np.sum((summed_spots.iloc[i], background[0] / 32), axis=0)
    combined_spots.iloc[i] = np.sum((combined_spots.iloc[i], background[0] / 32), axis=0)
#     phase_map.iloc[i] = np.sum((phase_map.iloc[i], background_counts / 32), axis=0)

#shift and define bins for energy-phase maps
phase_bins = np.linspace(0.0, 1-1/32, 32)
energy_bins = np.linspace(26, 300, 275)

summed_spots=np.array(summed_spots)
combined_spots=np.array(combined_spots)

shifted_summed_phase = np.concat((summed_spots[8:,:275],summed_spots[:8,:275]),axis=0) #18 for 300rs, [7/8,:700] for 600rs
shifted_combined_phase = np.concat((combined_spots[8:,:275],combined_spots[:8,:275]),axis=0) #18 for 300rs, [7/8,:700] for 600rs

### Plot bolometric lightcurve (or switch sum axes and change to energy_bins for comparison of E distrib)
summed_vals=shifted_summed_phase.sum(axis=1)
combined_vals=shifted_combined_phase.sum(axis=1)
plt.plot(phase_bins,summed_vals,label='Summed hotspots output')
plt.plot(phase_bins, combined_vals, alpha=0.5, label='Combined hotspots output')
plt.title('Bolometric LC')
plt.legend()
# plt.savefig(IMG_DIR / 'combined_bolometric.png')
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