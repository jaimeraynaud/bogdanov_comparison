import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_bounds(point,gridline):
    '''
    :param point: value within the grid
    :param gridline: array defining grid points along 1D
    :return: lower bound, upper bound, index of lower bound, index of upper bound
    '''

    if gridline[1]-gridline[0] > 0: #ascending grid
        for i in range(len(gridline) - 1):
            if gridline[i] <= point <= gridline[i + 1]:
                return gridline[i], gridline[i + 1], i, i + 1
    return 0,0,0,0


# specifications for response func
filename = "NICER_Apr2022_J0740_undershoot100_rsp.txt"
NrowrespmatNICER = 1558  # Nrows
NcolrespmatNICER = 304   # Ncolumns
eminNICER = np.zeros(NrowrespmatNICER, dtype=np.float64)
emaxNICER = np.zeros(NrowrespmatNICER, dtype=np.float64)
startchanNICER = np.zeros(NrowrespmatNICER, dtype=np.int32)
areaNICER = np.zeros((NrowrespmatNICER, NcolrespmatNICER - 3), dtype=np.float64)

# Read the file based on C snippet from Alex
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        values = line.strip().split()
        if len(values) < NcolrespmatNICER:
            raise ValueError(f"Line {i+1} has fewer than {NcolrespmatNICER} values")

        eminNICER[i] = float(values[0])
        emaxNICER[i] = float(values[1])
        startchanNICER[i] = int(values[2])
        areaNICER[i, :] = [float(x) for x in values[3:]]

#read in data produced from fortran code in units of photon counts cm^-2 s^-1
spot1_wendy = pd.read_csv('wendy_outputs/spot1_photcounts_NICERandXMM_v2.csv')
spot2_wendy = pd.read_csv('wendy_outputs/spot2_photcounts_NICERandXMM_v2.csv')

# #read in data for our method - files are already in final shape
# spot1_our_method = np.loadtxt('reproducing/spot1_photcounts_nofolding.dat')
# spot2_our_method = np.loadtxt('reproducing/spot2_photcounts_nofolding.dat')

spot1_our = np.loadtxt('reproducing/spot1_test_data_counts.dat')
spot2_our = np.loadtxt('reproducing/spot2_test_data_counts.dat')

# Save data as CSV with same name format
# np.savetxt('jaime_outputs/offset/spot1_photcounts_nofolding_jaime.csv', spot1_our_method, delimiter=',', fmt='%.10e')
# np.savetxt('jaime_outputs/offset/spot2_photcounts_nofolding_jaime.csv', spot2_our_method, delimiter=',', fmt='%.10e')

#read in provided data for J0740
j0740=pd.read_csv('j0740_phase_channel_model.txt', sep=' ',header=None)
background=np.array(j0740[5]).reshape(94,32) #counts due to background
NS_counts=np.array(j0740[4]).reshape(94,32) #counts from hotspots
best_fit=np.array(j0740[3]).reshape(94,32) #combined best fit to both NS + background counts
spot1_wendy=np.array(spot1_wendy)
spot2_wendy=np.array(spot2_wendy)

both_spots_wendy=spot1_wendy+spot2_wendy
both_spots_our_method=spot1_our+spot2_our

print("Jaime's data shape: ", both_spots_our_method.shape)
print("Wendy's data shape: ",both_spots_wendy.shape)

Nchan_wendy=len(both_spots_wendy[0])
Nchan_jaime=len(both_spots_our_method[0])
map_wendy=np.zeros((32,94))
map_jaime=np.zeros((32,94))

#fold through response and convert to units of instrument counts
chan_width=0.005 #in eV

# Process Wendy method
for i in range(32): #number of phases
    for j in range(Nchan_wendy): #number of channels
        val1,val2,i1,i2=find_bounds(j*chan_width+0.1+chan_width/2.0, eminNICER) #locate bounds for channel bin centers not edges (response func is unevenly binned)
        map_wendy[i,:] = map_wendy[i,:]+both_spots_wendy[i,j]*areaNICER[i1,30:124]*(1.0/32.0)*2733.81*1000.0 #30:124 = target chans, (1/32)*2733.81*1000.0 = exposure time per bin

# # Process Jaime method with the same transformations
# for i in range(32):
#     for j in range(Nchan_jaime):
#         val1,val2,i1,i2=find_bounds(j*chan_width+0.1+chan_width/2.0, eminNICER)
#         map_jaime[i,:] = map_jaime[i,:]+both_spots_our_method[i,j]*areaNICER[i1,30:124]*(1.0/32.0)*2733.81*1000.0


#plot energy-phase map
phase_bins = np.linspace(0.0, 1-1/32, 32)
energy_bins = np.linspace(31, 123, 94)
shifted_phase_wendy = np.concatenate((map_wendy[30:,],map_wendy[:30,]),axis=0) #shift phase to try and match (shift actual hotspots to fine tune)
shifted_phase_jaime = np.concatenate((both_spots_our_method[30:,],both_spots_our_method[:30,]),axis=0)
# shifted_phase_responsefunc = np.concatenate((both_spots_responsefunc[30:,],both_spots_responsefunc[:30,]),axis=0)

# Keep Wendy comparison for continuity
rel_diff=(NS_counts-shifted_phase_wendy[:32,:].T)/NS_counts

# # Jaime diagnostics against expected testcase data
# jaime_phase_map = shifted_phase_jaime[:32, :].T
# jaime_abs_diff = np.abs(NS_counts - jaime_phase_map)
# rel_diff_jaime = np.divide(
#     jaime_abs_diff,
#     NS_counts,
#     out=np.zeros_like(jaime_abs_diff),
#     where=NS_counts != 0
# )
# total_photons_jaime = np.sum(jaime_phase_map)
# normalized_rel_diff_jaime = np.divide(
#     jaime_abs_diff,
#     total_photons_jaime,
#     out=np.zeros_like(jaime_abs_diff),
#     where=total_photons_jaime != 0
# )

# plot = plt.pcolormesh(phase_bins, energy_bins, rel_diff, cmap='coolwarm', shading='nearest',vmin=-0.2,vmax=0.2) #plot rel diff map
# plt.xlabel('phase')
# plt.ylabel('energy channel')
# plt.colorbar(plot)
# plt.show()

# # Relative difference phase map for Jaime
# plot = plt.pcolormesh(phase_bins, energy_bins, rel_diff_jaime, cmap='coolwarm', shading='nearest')
# plt.title('Jaime Relative Difference Phase Map')
# plt.xlabel('phase')
# plt.ylabel('energy channel')
# plt.colorbar(plot)
# plt.show()

# # Normalized relative difference phase map for Jaime
# plot = plt.pcolormesh(phase_bins, energy_bins, normalized_rel_diff_jaime, cmap='coolwarm', shading='nearest')
# plt.title('Jaime Normalized Relative Difference Phase Map')
# plt.xlabel('phase')
# plt.ylabel('energy channel')
# plt.colorbar(plot)
# plt.show()

# # Scatter: relative difference vs expected testcase counts
# x_expected = NS_counts.flatten()
# y_rel = rel_diff_jaime.flatten()
# mask_rel = np.isfinite(x_expected) & np.isfinite(y_rel) & (x_expected > 0) & (y_rel > 0)
# plt.figure(figsize=(10, 6))
# plt.scatter(x_expected[mask_rel], y_rel[mask_rel], alpha=0.5, s=10)
# plt.xlabel('Expected Testcase Counts')
# plt.ylabel('Relative Difference')
# plt.title('Jaime Relative Difference vs Expected Counts')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # Scatter: normalized relative difference vs expected testcase counts
# y_norm_rel = normalized_rel_diff_jaime.flatten()
# mask_norm = np.isfinite(x_expected) & np.isfinite(y_norm_rel) & (x_expected > 0) & (y_norm_rel > 0)
# plt.figure(figsize=(10, 6))
# plt.scatter(x_expected[mask_norm], y_norm_rel[mask_norm], alpha=0.5, s=10)
# plt.xlabel('Expected Testcase Counts')
# plt.ylabel('Normalized Relative Difference')
# plt.title('Jaime Normalized Relative Difference vs Expected Counts')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

#bolometric lightcurve comparison
summed_vals_wendy=shifted_phase_wendy.sum(axis=1)
summed_vals_our_method=shifted_phase_jaime.sum(axis=1)*((1.0/32.0)*2733.81*1000.0)/((1.0/32.0)*(10.0**6)) #apply same folding to response func data for fair comparison
# summed_vals_responsefunc=shifted_phase_responsefunc.sum(axis=1)*((1.0/32.0)*2733.81*1000.0)/((1.0/32.0)*(10.0**6)) #apply same folding to response func data for fair comparison
summed_vals_expected=NS_counts.sum(axis=0)
print(phase_bins)
phase_bins_ck=(phase_bins-1/64.)#%1.0
print(phase_bins_ck)

fig, ax = plt.subplots()
ax.plot(phase_bins_ck, summed_vals_wendy, label='Wendys output')
ax.plot(phase_bins, summed_vals_our_method, label='Our output')
ax.plot(phase_bins, summed_vals_expected, label='Dittmans output')
ax.set_title("J0740 Bolometric LC")
ax.legend()
fig.tight_layout()
# fig.savefig("/Users/jraynau1/Workspace/plots/j0740_bolometric.png", dpi=300)
plt.show()

# # Quarter-channel lightcurve comparison (sum each quarter of axis=1 channels)
# channel_splits = np.array_split(np.arange(shifted_phase_wendy.shape[1]), 4)
# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
#
# for idx, channel_idx in enumerate(channel_splits):
#     ax = axes.flat[idx]
#     start_idx = channel_idx[0]
#     end_idx = channel_idx[-1]
#
#     wendy_subset = shifted_phase_wendy[:, channel_idx].sum(axis=1)
#     jaime_subset = shifted_phase_jaime[:, channel_idx].sum(axis=1)
#     expected_subset = NS_counts[channel_idx, :].sum(axis=0)
#
#     ax.plot(phase_bins_ck, wendy_subset, label='wendy method')
#     ax.plot(phase_bins, jaime_subset, label='our method')
#     ax.plot(phase_bins, expected_subset, label='expected testcase')
#     ax.set_title(f'Channels {31 + start_idx}-{31 + end_idx}')
#     ax.set_xlabel('phase')
#     ax.set_ylabel('summed counts')
#     ax.grid(True, alpha=0.3)
#     if idx == 0:
#         ax.legend()
#
# fig.suptitle('Bolometric Lightcurves by Channel Quartile', y=1.02)
# plt.tight_layout()
# plt.show()

