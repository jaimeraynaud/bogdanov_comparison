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
filename = "/Users/xingf/PycharmProjects/nn_mcmc_analysis/data/phase_maps/J0740_data/NICER_Apr2022_J0740_undershoot100_rsp.txt"
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
spot1 = pd.read_csv('/Users/xingf/PycharmProjects/nn_mcmc_analysis/data/phase_maps/spot1_photcounts_NICERandXMM_v2.csv')
spot2 = pd.read_csv('/Users/xingf/PycharmProjects/nn_mcmc_analysis/data/phase_maps/spot2_photcounts_NICERandXMM_v2.csv')
#read in provided data for J0740
j0740=pd.read_csv('/Users/xingf/PycharmProjects/nn_mcmc_analysis/data/phase_maps/J0740_data/j0740_phase_channel_model.txt', sep=' ',header=None)
background=np.array(j0740[5]).reshape(94,32) #counts due to background
NS_counts=np.array(j0740[4]).reshape(94,32) #counts from hotspots
best_fit=np.array(j0740[3]).reshape(94,32) #combined best fit to both NS + background counts
spot1=np.array(spot1)
spot2=np.array(spot2)
both_spots=spot1+spot2
Nchan=len(both_spots[0])
map=np.zeros((32,94))

#fold through response and convert to units of instrument counts
chan_width=0.005 #in eV
for i in range(32): #number of phases
    for j in range(Nchan):
        val1,val2,i1,i2=find_bounds(j*chan_width+0.1+chan_width/2.0, eminNICER) #locate bounds for channel bin centers not edges (response func is unevenly binned)
        map[i,:] = map[i,:]+both_spots[i,j]*areaNICER[i1,30:124]*(1.0/32.0)*2733.81*1000.0 #30:124 = target chans, (1/32)*2733.81*1000.0 = exposure time per bin

#plot energy-phase map
phase_bins = np.linspace(0.0, 1-1/32, 32)
energy_bins = np.linspace(31, 123, 94)
shifted_phase= np.concat((map[30:,],map[:30,]),axis=0) #shift phase to try and match (shift actual hotspots to fine tune)
rel_diff=(NS_counts-shifted_phase[:32,:].T)/NS_counts
plot = plt.pcolormesh(phase_bins, energy_bins, shifted_phase.T, cmap='magma', shading='nearest') #plot map
# plot = plt.pcolormesh(phase_bins, energy_bins, rel_diff, cmap='coolwarm', shading='nearest',vmin=-0.2,vmax=0.2) #plot rel diff map
plt.xlabel('phase')
plt.ylabel('energy channel')
plt.colorbar(plot)
plt.show()

#bolometric lightcurve comparison
summed_vals=shifted_phase.sum(axis=1)
summed_vals2=NS_counts.sum(axis=0)
plt.plot(phase_bins,summed_vals,label='our method')
plt.plot(phase_bins,summed_vals2,label='expected testcase')
plt.legend()
plt.show()
