import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Usage: python plot_photon_hits.py path/to/output_dir
outdir = Path(sys.argv[1]) if len(sys.argv)>1 else Path('output_data')
all_file = outdir / 'photon_hits_all.dat'
spot_file = outdir / 'photon_hits_spot3.dat'

# Load files (skip header lines beginning with '#')
def load_data(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            if line.strip()=='' or line.lstrip().startswith('#'):
                continue
            parts=line.split()
            if len(parts) < 4:
                continue
            a0=float(parts[0]); b0=float(parts[1]); x=float(parts[2]); y=float(parts[3])
            data.append((a0,b0,x,y))
    return np.array(data) if data else np.zeros((0,4))

all_data = load_data(all_file)
spot_data = load_data(spot_file)

fig, axes = plt.subplots(1,2, figsize=(12,6))
# Left: image-plane coords (alpha0,beta0)
if all_data.size>0:
    axes[0].scatter(all_data[:,0], all_data[:,1], s=1, color='gray', label='all hits')
if spot_data.size>0:
    axes[0].scatter(spot_data[:,0], spot_data[:,1], s=2, color='red', label='hotspot hits')
axes[0].set_xlabel('alpha0')
axes[0].set_ylabel('beta0')
axes[0].set_aspect('equal', 'box')
axes[0].legend()
axes[0].set_title('Image-plane coordinates (alpha0,beta0)')

# Right: surface x,y
if all_data.size>0:
    axes[1].scatter(all_data[:,2], all_data[:,3], s=1, color='gray', label='all hits')
if spot_data.size>0:
    axes[1].scatter(spot_data[:,2], spot_data[:,3], s=2, color='red', label='hotspot hits')
axes[1].set_xlabel('x_surf (m)')
axes[1].set_ylabel('y_surf (m)')
axes[1].set_aspect('equal', 'box')
axes[1].legend()
axes[1].set_title('Stellar-surface Cartesian (x,y)')

plt.tight_layout()
plt.show()