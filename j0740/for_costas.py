import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# ------------------------------------------------------------------
# Load J0740 phase-channel model file
# Column layout (0-indexed):
#   0 -> channel number
#   1 -> rotational phase bin (0..31)
#   2 -> observed counts (data)
#   3 -> best-fit model counts (total = NS surface + background)
#   4 -> best-fit NS-surface-only component
#   5 -> best-fit phase-independent background
# ------------------------------------------------------------------

j0740 = pd.read_csv('j0740_phase_channel_model.txt', sep=' ', header=None)

observed   = np.array(j0740[2]).reshape(94, 32)
best_fit   = np.array(j0740[3]).reshape(94, 32)
ns_model   = np.array(j0740[4]).reshape(94, 32)
background = np.array(j0740[5]).reshape(94, 32)

observed_with_bkg    = observed
observed_without_bkg = observed - background
bestfit_with_bkg     = best_fit
bestfit_without_bkg  = ns_model

# ------------------------------------------------------------------
# Sum over energy axis (axis=0, since shape is (94 energy, 32 phase))
# ------------------------------------------------------------------
phase_bins = np.linspace(0.0, 1 - 1 / 32, 32)

obs_lc_with_bkg    = observed_with_bkg.sum(axis=0)
obs_lc_without_bkg = observed_without_bkg.sum(axis=0)
bf_lc_with_bkg     = bestfit_with_bkg.sum(axis=0)
bf_lc_without_bkg  = bestfit_without_bkg.sum(axis=0)

# ------------------------------------------------------------------
# Load Alamos light curve from j0740_counts_plus_background.dat
# ------------------------------------------------------------------
alamos_data = np.loadtxt('j0740_counts_plus_background.dat')

if alamos_data.ndim == 1:
    if alamos_data.size != 32:
        raise ValueError(
            f"Expected 32 phase bins in Alamos file, got {alamos_data.size}"
        )
    alamos_lc = alamos_data
elif alamos_data.ndim == 2:
    if alamos_data.shape[1] == 32:
        alamos_lc = alamos_data.sum(axis=0)
    elif alamos_data.shape[0] == 32:
        alamos_lc = alamos_data.sum(axis=1)
    else:
        raise ValueError(
            f"Could not find a length-32 phase axis in Alamos data of shape {alamos_data.shape}"
        )
else:
    raise ValueError(f"Unexpected Alamos data ndim={alamos_data.ndim}")

print(f"Loaded Alamos light curve with shape {alamos_data.shape} -> reduced to length {alamos_lc.size}")

# ------------------------------------------------------------------
# Save ONLY phase and observed counts (with background) to CSV
# ------------------------------------------------------------------
output_path = 'j0740_bolometric_with_background.csv'
df_out = pd.DataFrame({
    'phase': phase_bins,
    'observed_counts': obs_lc_with_bkg,
})
df_out.to_csv(output_path, index=False)
print(f"Saved observed bolometric light curve (with background) to {output_path}")

# ------------------------------------------------------------------
# Re-load observed counts from the CSV we just wrote, so the plotted
# observed line comes from this file rather than the in-memory array.
# ------------------------------------------------------------------
observed_from_csv = pd.read_csv(output_path)
phase_from_csv    = observed_from_csv['phase'].to_numpy()
obs_lc_from_csv   = observed_from_csv['observed_counts'].to_numpy()


def plot_lc(bestfit_lc, alamos_lc, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(phase_from_csv, obs_lc_from_csv,
            label='Observed counts', marker='o', markersize=4)
    ax.plot(phase_bins, bestfit_lc,
            label='Best-fit model counts', marker='s', markersize=4)
    ax.plot(phase_bins, alamos_lc,
            label='Alamos', marker='^', markersize=4)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Counts (summed over energy)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# Figure 1: WITH background
fig1 = plot_lc(
    bf_lc_with_bkg,
    alamos_lc,
    "J0740 Bolometric Light Curve (with background)",
)

# Figure 2: WITHOUT background
fig2 = plot_lc(
    bf_lc_without_bkg,
    alamos_lc,
    "J0740 Bolometric Light Curve (background subtracted)",
)

plt.show()