"""
plot_j0030_lightcurve_comparison.py

Compares our Fortran EPhase_map output (j0030_test.dat) against
background-subtracted observed counts from j0030_phase_channel_model.txt.

j0030_phase_channel_model.txt column layout (0-based):
    col 0 : channel number            (energy axis)
    col 1 : rotational phase bin
    col 2 : observed counts           ← we use this
    col 3 : best-fit model counts
    col 4 : NS-surface-only model counts
    col 5 : phase-independent background

A constant background of BACKGROUND counts is subtracted from the
energy-summed 1D light curve (not from the 2D energy-phase map).

Fortran write format:
    write(43, '(*(G20.12))') ((EPhase_map(i,k), k=1, real_nchan_cfg), i=1, 32)
    → flat values, phase-major: all energies for phase 1, then phase 2, …
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# MODEL_FILE = Path(
#     # "/Users/jraynau1/Workspace/CLionProjects/ray_tracing_training_nn"
#     # "/output_data/j0030_test_600.dat"
# )

MODEL_FILE = Path(__file__).parent / "data" / "j0030_test.dat"
OBS_FILE = Path(__file__).parent / "data" / "j0030_phase_channel_model.txt"

OUTPUT_DIR = Path(
    "/Users/jraynau1/Workspace/data_analysis_visualization/training_nn/plots"
)

# ---------------------------------------------------------------------------
# Constants — adjust REAL_NCHAN to match your Fortran real_nchan_cfg
# ---------------------------------------------------------------------------
NPHASE_MODEL = 32     # i = 1..32  (Fortran phase bins)
REAL_NCHAN   = 260    # k = 1..real_nchan_cfg  ← change if token count mismatches

# Column in j0030_phase_channel_model.txt (0-based)
OBS_COL    = 2        # observed counts
BKG_COL    = 5

BACKGROUND = 2.4*13500    # constant background subtracted from the 1D light curve


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_phase_channel_table(path: Path) -> np.ndarray:
    """Load the j0030_phase_channel_model.txt table."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(
            f"Expected a 2D table with at least 6 columns, got shape {data.shape}"
        )
    return data


def infer_phase_energy_map(data: np.ndarray,
                           value_column: int,
                           background_column: int | None = None,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape the table into an (n_energy, n_phase) map.

    col 0 = channel (energy axis), col 1 = phase bin.

    Parameters
    ----------
    data             : full table loaded by load_phase_channel_table
    value_column     : column index whose values fill the map
    background_column: if given, subtract this column from value_column
                       before reshaping

    Returns
    -------
    energy_vals : (n_energy,)
    phase_vals  : (n_phase,)
    map_ep      : (n_energy, n_phase)
    """
    energy_vals = np.unique(data[:, 0])
    phase_vals  = np.unique(data[:, 1])

    order       = np.lexsort((data[:, 1], data[:, 0]))
    sorted_data = data[order]

    values = sorted_data[:, value_column].copy()
    if background_column is not None:
        values -= sorted_data[:, background_column]

    map_ep = values.reshape(len(energy_vals), len(phase_vals))  # (n_energy, n_phase)
    return energy_vals, phase_vals, map_ep


def load_fortran_ephase_map(filepath: Path,
                            nphase: int = NPHASE_MODEL,
                            real_nchan: int = REAL_NCHAN) -> np.ndarray:
    """
    Load the flat Fortran output and reshape to (nphase, real_nchan).

    Fortran write order: ((EPhase_map(i,k), k=1,Nchan), i=1,32)
    → phase-major: for each phase i, all energies k.
    Returns array shape (nphase, real_nchan).
    """
    flat = np.loadtxt(filepath).ravel()
    n_expected = nphase * real_nchan
    if flat.size != n_expected:
        if flat.size % nphase == 0:
            inferred = flat.size // nphase
            raise ValueError(
                f"Expected {n_expected} values ({nphase} × {real_nchan}), "
                f"got {flat.size}.\n"
                f"  → Try setting REAL_NCHAN = {inferred}"
            )
        raise ValueError(
            f"Expected {n_expected} values, got {flat.size}. "
            f"Check NPHASE_MODEL and REAL_NCHAN."
        )
    return flat.reshape(nphase, real_nchan)    # (nphase, real_nchan)


# ---------------------------------------------------------------------------
# Figure 1 — Bolometric light curves
# ---------------------------------------------------------------------------
def plot_lightcurves(our_map:    np.ndarray,
                     obs_map_ep: np.ndarray,
                     obs_phase:  np.ndarray,
                     output_dir: Path,
                     cycles:     int = 2,
                     show:       bool = True) -> None:
    """
    Plot energy-summed light curves for our model and the
    background-subtracted observed counts.

    our_map    : (nphase, real_nchan)  — Fortran output
    obs_map_ep : (n_energy, n_phase)   — raw observed counts (col 2)
    obs_phase  : (n_phase,)            — phase bin values from the table
    cycles     : how many pulse cycles to show

    Background is subtracted from the 1D light curve after energy summation.
    """
    def fractional_circular_shift(values: np.ndarray,
                                  shift_bins: float,
                                  axis: int = -1) -> np.ndarray:
        n     = values.shape[axis]
        freqs = np.fft.rfft(values, axis=axis)
        k     = np.arange(freqs.shape[axis])
        phase = np.exp(2j * np.pi * k * shift_bins / n)
        shape = [1] * values.ndim
        shape[axis] = phase.size
        phase = phase.reshape(shape)
        return np.fft.irfft(freqs * phase, n=n, axis=axis)

    our_map = fractional_circular_shift(our_map, shift_bins=22.5, axis=0)
    our_lc  = our_map.sum(axis=1)                    # (nphase,)
    obs_lc  = obs_map_ep.sum(axis=0)                 # (n_phase,) — already background-subtracted upstream


# Build phase axes — treat integer bin indices as bin centres
    our_base = (np.arange(NPHASE_MODEL) + 0.5) / NPHASE_MODEL
    if np.allclose(obs_phase, np.round(obs_phase)):
        obs_base = (obs_phase + 0.5) / len(obs_phase)
    else:
        obs_base = obs_phase

    # Tile for multiple cycles
    our_phase_ax = np.concatenate([our_base + k for k in range(cycles)])
    obs_phase_ax = np.concatenate([obs_base + k for k in range(cycles)])
    our_lc_plot  = np.tile(our_lc, cycles)
    obs_lc_plot  = np.tile(obs_lc, cycles)

    print(f"  Our model light curve      — sum={our_lc.sum():.4e}, "
          f"min={our_lc.min():.4e}, max={our_lc.max():.4e}")
    print(f"  Observed (bkg-subtracted)  — sum={obs_lc.sum():.4e}, "
          f"min={obs_lc.min():.4e}, max={obs_lc.max():.4e}")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)

    ax.plot(obs_phase_ax, obs_lc_plot,
            color="tab:blue", lw=2.2,
            label=f"Observed  (col {OBS_COL} − background col {BKG_COL}, summed)")
    ax.plot(our_phase_ax, our_lc_plot,
            color="tab:orange", lw=2.0, ls="--",
            label=f"Our model  ({MODEL_FILE.name})")

    ax.set_title("J0030: energy-summed light curve comparison")
    ax.set_xlabel("phase")
    ax.set_ylabel("counts  (background subtracted)")
    ax.set_xlim(our_phase_ax.min(), our_phase_ax.max())
    ax.legend(frameon=False)
    fig.tight_layout()

    out = output_dir / "j0030_lightcurve_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved → {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — 2D energy-phase maps
# ---------------------------------------------------------------------------
def plot_2d_maps(our_map:    np.ndarray,
                 obs_map_ep: np.ndarray,
                 obs_energy: np.ndarray,
                 obs_phase:  np.ndarray,
                 output_dir: Path,
                 cycles:     int = 1,
                 show:       bool = True) -> None:
    """
    Side-by-side 2D energy-phase maps.

    our_map    : (nphase, real_nchan)  → transposed to (real_nchan, nphase)
    obs_map_ep : (n_energy, n_phase)   — raw observed counts (col 2, no bkg subtraction)
    """
    # Our map: transpose to (energy, phase)
    our_ep         = our_map.T                               # (real_nchan, nphase)
    our_chan_axis  = np.arange(our_ep.shape[0])
    our_phase_axis = (np.arange(NPHASE_MODEL) + 0.5) / NPHASE_MODEL

    # Observed phase axis
    if np.allclose(obs_phase, np.round(obs_phase)):
        obs_phase_axis = (obs_phase + 0.5) / len(obs_phase)
    else:
        obs_phase_axis = obs_phase

    # Tile for multiple cycles
    if cycles > 1:
        our_phase_axis = np.concatenate([our_phase_axis + k for k in range(cycles)])
        obs_phase_axis = np.concatenate([obs_phase_axis + k for k in range(cycles)])
        our_ep         = np.tile(our_ep,     (1, cycles))
        obs_map_ep     = np.tile(obs_map_ep, (1, cycles))

    vmin     = 0.0
    vmax_our = float(np.percentile(our_ep[our_ep > 0],         99)) if np.any(our_ep > 0)       else 1.0
    vmax_obs = float(np.percentile(obs_map_ep[obs_map_ep > 0], 99)) if np.any(obs_map_ep > 0)   else 1.0

    print(f"  2D map colour scale — our vmax={vmax_our:.3e}, "
          f"observed vmax={vmax_obs:.3e}")

    fig = plt.figure(figsize=(16, 5), dpi=180)
    gs  = gridspec.GridSpec(1, 4,
                            width_ratios=[1, 0.04, 1, 0.04],
                            wspace=0.08)
    ax_our  = fig.add_subplot(gs[0])
    cax_our = fig.add_subplot(gs[1])
    ax_obs  = fig.add_subplot(gs[2])
    cax_obs = fig.add_subplot(gs[3])

    # Our model
    mesh_our = ax_our.pcolormesh(
        our_phase_axis, our_chan_axis, our_ep,
        shading="auto", cmap="magma",
        vmin=vmin, vmax=vmax_our,
    )
    ax_our.set_title(f"Our model  ({MODEL_FILE.name})", fontsize=10)
    ax_our.set_xlabel("phase")
    ax_our.set_ylabel("energy channel index")
    ax_our.set_xlim(our_phase_axis.min(), our_phase_axis.max())
    fig.colorbar(mesh_our, cax=cax_our, label="counts")

    # Raw observed counts (no background subtraction on 2D map)
    mesh_obs = ax_obs.pcolormesh(
        obs_phase_axis, obs_energy, obs_map_ep,
        shading="auto", cmap="magma",
        vmin=vmin, vmax=vmax_obs,
    )
    ax_obs.set_title(
        f"Observed  (col {OBS_COL} − background col {BKG_COL})", fontsize=10
    )
    ax_obs.set_xlabel("phase")
    ax_obs.set_ylabel("energy channel")
    ax_obs.set_xlim(obs_phase_axis.min(), obs_phase_axis.max())
    fig.colorbar(mesh_obs, cax=cax_obs, label="counts")

    fig.suptitle(
        f"J0030 phase–energy map comparison  ({cycles} cycle{'s' if cycles > 1 else ''})",
        fontsize=11,
    )
    fig.tight_layout()

    out = output_dir / "j0030_2d_map_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved → {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading model   : {MODEL_FILE}")
    print(f"Loading observed: {OBS_FILE}")
    print(f"  Using col {OBS_COL} (observed counts) with background column {BKG_COL} subtracted in both 2D map and 1D light curve")
    # --- Our Fortran output ---
    our_map = load_fortran_ephase_map(MODEL_FILE,
                                      nphase=NPHASE_MODEL,
                                      real_nchan=REAL_NCHAN)
    print(f"\n  Our EPhase_map shape : {our_map.shape}  "
          f"(nphase={NPHASE_MODEL}, real_nchan={REAL_NCHAN})")
    print(f"  Our sum              : {our_map.sum():.4e}")
    print(f"  Our min / max        : {our_map.min():.4e} / {our_map.max():.4e}")

    # --- Raw observed counts (background subtraction happens in plot_lightcurves) ---
    data = load_phase_channel_table(OBS_FILE)
    obs_energy, obs_phase, obs_map_ep = infer_phase_energy_map(
        data,
        value_column=OBS_COL,
        background_column=BKG_COL,
    )
    print(f"\n  Observed map shape   : {obs_map_ep.shape}  "
          f"(n_energy={len(obs_energy)}, n_phase={len(obs_phase)})")
    print(f"  Observed sum (bkg-sub) : {obs_map_ep.sum():.4e}")
    print(f"  Observed min / max     : {obs_map_ep.min():.4e} / {obs_map_ep.max():.4e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: bolometric light curves
    print("\n--- Light curve comparison ---")
    plot_lightcurves(
        our_map    = our_map,
        obs_map_ep = obs_map_ep,
        obs_phase  = obs_phase,
        output_dir = OUTPUT_DIR,
        cycles     = 2,
        show       = True,
    )

    # Figure 2: 2D energy-phase maps
    print("\n--- 2D map comparison ---")
    plot_2d_maps(
        our_map    = our_map,
        obs_map_ep = obs_map_ep,
        obs_energy = obs_energy,
        obs_phase  = obs_phase,
        output_dir = OUTPUT_DIR,
        cycles     = 1,
        show       = True,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()