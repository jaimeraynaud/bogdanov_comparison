import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def find_bounds(point, gridline):
    '''
    :param point: value within the grid
    :param gridline: array defining grid points along 1D
    :return: lower bound, upper bound, index of lower bound, index of upper bound
    '''
    if gridline[1] - gridline[0] > 0:  # ascending grid
        for i in range(len(gridline) - 1):
            if gridline[i] <= point <= gridline[i + 1]:
                return gridline[i], gridline[i + 1], i, i + 1
    return 0, 0, 0, 0


# specifications for response func
filename = "NICER_Apr2022_J0740_undershoot100_rsp.txt"
NrowrespmatNICER = 1558
NcolrespmatNICER = 304
eminNICER = np.zeros(NrowrespmatNICER, dtype=np.float64)
emaxNICER = np.zeros(NrowrespmatNICER, dtype=np.float64)
startchanNICER = np.zeros(NrowrespmatNICER, dtype=np.int32)
areaNICER = np.zeros((NrowrespmatNICER, NcolrespmatNICER - 3), dtype=np.float64)

with open(filename, 'r') as f:
    for i, line in enumerate(f):
        values = line.strip().split()
        if len(values) < NcolrespmatNICER:
            raise ValueError(f"Line {i+1} has fewer than {NcolrespmatNICER} values")
        eminNICER[i] = float(values[0])
        emaxNICER[i] = float(values[1])
        startchanNICER[i] = int(values[2])
        areaNICER[i, :] = [float(x) for x in values[3:]]

# read in data produced from fortran code in units of photon counts cm^-2 s^-1
spot1_wendy = pd.read_csv('wendy_outputs/spot1_photcounts_NICERandXMM_v2.csv')
spot2_wendy = pd.read_csv('wendy_outputs/spot2_photcounts_NICERandXMM_v2.csv')

spot1_old = np.loadtxt('reproducing/spot1_test_data_counts.dat')
spot2_old = np.loadtxt('reproducing/spot2_test_data_counts.dat')

spot1_our = np.loadtxt('latest/spot1_test_data_counts.dat')
spot2_our = np.loadtxt('latest/spot2_test_data_counts.dat')

# read in provided data for J0740
j0740 = pd.read_csv('j0740_phase_channel_model.txt', sep=' ', header=None)
background = np.array(j0740[5]).reshape(94, 32)
NS_counts = np.array(j0740[4]).reshape(94, 32)
best_fit = np.array(j0740[3]).reshape(94, 32)
spot1_wendy = np.array(spot1_wendy)
spot2_wendy = np.array(spot2_wendy)

both_spots_wendy = spot1_wendy + spot2_wendy
both_spots_our_method = spot1_our + spot2_our
both_spots_old = spot1_old + spot2_old

print("Old's data shape: ", both_spots_old.shape)
print("New's data shape: ", both_spots_our_method.shape)
print("Wendy's data shape: ", both_spots_wendy.shape)

Nchan_wendy = len(both_spots_wendy[0])
Nchan_jaime = len(both_spots_our_method[0])
map_wendy = np.zeros((32, 94))
map_jaime = np.zeros((32, 94))

chan_width = 0.005  # in eV

# Process Wendy method
for i in range(32):
    for j in range(Nchan_wendy):
        val1, val2, i1, i2 = find_bounds(j * chan_width + 0.1 + chan_width / 2.0, eminNICER)
        map_wendy[i, :] = map_wendy[i, :] + both_spots_wendy[i, j] * areaNICER[i1, 30:124] * (1.0 / 32.0) * 2733.81 * 1000.0

# plot energy-phase map
phase_bins = np.linspace(0.0, 1 - 1 / 32, 32)
energy_bins = np.linspace(31, 123, 94)
shifted_phase_wendy = np.concatenate((map_wendy[30:, ], map_wendy[:30, ]), axis=0)
shifted_phase_jaime = np.concatenate((both_spots_our_method[30:, ], both_spots_our_method[:30, ]), axis=0)
shifted_phase_old = np.concatenate((both_spots_old[30:, ], both_spots_old[:30, ]), axis=0)

# our_method_2d: shape (energy=94, phase=32)
our_method_2d = shifted_phase_jaime[:32, :].T

# ------------------------------------------------------------------
# 2D Energy-Phase Map (Our model)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(
    phase_bins, energy_bins, our_method_2d,
    cmap='viridis', shading='nearest',
)
ax.set_xlabel("Phase")
ax.set_ylabel("Energy channel")
fig.colorbar(im, ax=ax, label="Photon Counts")
fig.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Bolometric lightcurve comparison
# ------------------------------------------------------------------
summed_vals_our_method = shifted_phase_jaime.sum(axis=1)
summed_vals_expected = NS_counts.sum(axis=0)

fig, ax = plt.subplots()
ax.plot(phase_bins, summed_vals_our_method, label='Our model')
ax.plot(phase_bins, summed_vals_expected, label='Dittman et al. model')
ax.set_xlabel("Phase")
ax.set_ylabel("Photon Counts")
ax.set_title("J0740 Bolometric LC")
ax.legend()
fig.tight_layout()
plt.show()