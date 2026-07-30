import numpy as np
import matplotlib.pyplot as plt

outdir = '/Users/jraynau1/Workspace/CLionProjects/ray_tracing/output_data/j0030/lmop150/'
all_file = outdir + 'photon_hits_all.dat'
spot_file = outdir + 'photon_hits_spot3.dat'

def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() == '' or line.lstrip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            th = float(parts[0]); ph = float(parts[1]); a0 = float(parts[2]); b0 = float(parts[3])
            data.append((th, ph, a0, b0))
    return np.array(data) if data else np.zeros((0, 4))

all_data = load_data(all_file)
spot_data = load_data(spot_file)

fig, ax = plt.subplots(figsize=(8, 8))

# Surface x,y (Cartesian projection) - converted from th/ph
R = 13466.0  # stellar radius in meters

if all_data.size > 0:
    th_all = all_data[:, 0]
    ph_all = all_data[:, 1]
    a0_all = all_data[:, 2]
    b0_all = all_data[:, 3]
    x_all = R * np.sin(th_all) * np.cos(ph_all)
    y_all = R * np.sin(th_all) * np.sin(ph_all)
    ax.scatter(a0_all, b0_all, s=1, color='gray', label='all hits')

if spot_data.size > 0:
    th_spot = spot_data[:, 0]
    ph_spot = spot_data[:, 1]
    a0_spot = spot_data[:, 2]
    b0_spot = spot_data[:, 3]
    x_spot = R * np.sin(th_spot) * np.cos(ph_spot)
    y_spot = R * np.sin(th_spot) * np.sin(ph_spot)
    ax.scatter(a0_spot, b0_spot, s=3, color='blue', alpha=0.5, label='hotspot (th/ph → x/y)')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_aspect('equal', 'box')
ax.legend(loc='upper right')
ax.set_title('Stellar-surface coordinates projected to (x, y) - blue: hotspot')

plt.tight_layout()
plt.show()