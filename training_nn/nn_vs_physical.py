from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

# Photon counts (2D energy-phase maps)
COUNTS_PHYSICAL_FILE = DATA_DIR / "j0030_test_physical.dat"
COUNTS_NN_FILE       = DATA_DIR / "j0030_test_nn.dat"

# MV211 fluxes (idistrib; 2D energy-phase maps)
FLUX_PHYSICAL_FILE   = DATA_DIR / "j0030_idistrib_physical.dat"
FLUX_NN_FILE         = DATA_DIR / "j0030_idistrib_nn.dat"

# Output directory for figures
OUTPUT_DIR = Path(__file__).parent / "plots"

# ---------------------------------------------------------------------------
# Constants — adjust REAL_NCHAN to match your Fortran real_nchan_cfg
# ---------------------------------------------------------------------------
NPHASE_MODEL       = 32    # i = 1..32  (Fortran phase bins)
REAL_NCHAN_COUNTS  = 260   # channels for photon counts files
REAL_NCHAN_FLUX    = 260   # channels for flux (idistrib) files

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_fortran_phase_major(filepath: Path,
                             nphase: int,
                             real_nchan: int) -> np.ndarray:
    """
    Load a flat Fortran output and reshape to (nphase, real_nchan).

    sql

    Copy code
    Fortran write order: ((Map(i,k), k=1,Nchan), i=1,Nphase)
    → phase-major: for each phase i, all energies k.
    """
    flat = np.loadtxt(filepath).ravel()
    n_expected = nphase * real_nchan
    if flat.size != n_expected:
        if flat.size % nphase == 0:
            inferred = flat.size // nphase
            raise ValueError(
                f"{filepath.name}: Expected {n_expected} values ({nphase}×{real_nchan}), "
                f"got {flat.size}. Try setting REAL_NCHAN = {inferred}"
            )
        raise ValueError(
            f"{filepath.name}: Expected {n_expected} values, got {flat.size}. "
            f"Check NPHASE_MODEL and REAL_NCHAN_*."
        )
    arr = flat.reshape(nphase, real_nchan)  # (nphase, real_nchan)
    return arr

def to_energy_phase(map_phase_nchan: np.ndarray) -> np.ndarray:
    """
    Convert shape (nphase, nchan) to (n_energy, n_phase) for plotting.
    """
    return map_phase_nchan.T  # (n_energy, n_phase)

def robust_limits(a: np.ndarray, allow_negative: bool = False) -> tuple[float, float]:
    """
    Choose colour scale limits robustly using percentiles.
    For count-like arrays (non-negative), vmin=0 and vmax=99th percentile of positives.
    If allow_negative=True, use symmetric limits around 0 based on 99th percentile of |a|.
    """
    finite = np.isfinite(a)
    if not np.any(finite):
        return (0.0, 1.0)


    vals = a[finite]
    if allow_negative:
        p = np.percentile(np.abs(vals), 99.0)
        p = float(p) if p > 0 else float(np.max(np.abs(vals))) if np.any(vals) else 1.0
        return (-p, p)

    positives = vals[vals > 0]
    if positives.size > 0:
        vmax = float(np.percentile(positives, 99.0))
        vmax = vmax if vmax > 0 else float(np.max(positives))
        return (0.0, vmax if vmax > 0 else 1.0)
    # fallback for all non-positives
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmin == vmax:
        vmax = vmin + 1.0
    return (vmin, vmax)

def relative_difference(nn: np.ndarray, phys: np.ndarray) -> np.ndarray:
    """
    Compute fractional difference (nn - phys) / denom, with a small epsilon to avoid division by zero.
    """
    eps = max(1e-12, 1e-6 * float(np.max(np.abs(phys))) if np.any(phys) else 1.0)
    denom = np.where(np.abs(phys) > eps, phys, np.sign(phys) * eps + (np.abs(phys) <= eps) * eps)
    return (nn - phys) / denom

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_2d_comparison(ep_phys: np.ndarray,
                       ep_nn: np.ndarray,
                       title_prefix: str,
                       outfile: Path) -> None:
    """
    Create a 3-panel figure: physical, neural network, and relative difference.
    ep_* are (n_energy, n_phase) arrays.
    """
    rel = relative_difference(ep_nn, ep_phys)

    vmin, vmax = robust_limits(ep_phys, allow_negative=False)
    vmin_nn, vmax_nn = robust_limits(ep_nn, allow_negative=False)
    vmin_rel, vmax_rel = robust_limits(rel, allow_negative=True)

    n_energy, n_phase = ep_phys.shape
    phase_axis = (np.arange(n_phase) + 0.5) / n_phase
    energy_axis = np.arange(n_energy)

    fig = plt.figure(figsize=(16, 4.8), dpi=180)
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05], wspace=0.15)

    ax1 = fig.add_subplot(gs[0])
    cax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    cax2 = fig.add_subplot(gs[3])
    ax3 = fig.add_subplot(gs[4])
    cax3 = fig.add_subplot(gs[5])

    m1 = ax1.pcolormesh(phase_axis, energy_axis, ep_phys, shading="auto", cmap="magma",
                        vmin=vmin, vmax=vmax)
    ax1.set_title(f"{title_prefix}: Physical")
    ax1.set_xlabel("phase")
    ax1.set_ylabel("energy channel")
    fig.colorbar(m1, cax=cax1, label="counts" if "Counts" in title_prefix else "flux")

    m2 = ax2.pcolormesh(phase_axis, energy_axis, ep_nn, shading="auto", cmap="magma",
                        vmin=vmin_nn, vmax=vmax_nn)
    ax2.set_title(f"{title_prefix}: Neural Network")
    ax2.set_xlabel("phase")
    ax2.set_ylabel("energy channel")
    fig.colorbar(m2, cax=cax2, label="counts" if "Counts" in title_prefix else "flux")

    m3 = ax3.pcolormesh(phase_axis, energy_axis, rel, shading="auto", cmap="coolwarm",
                        vmin=vmin_rel, vmax=vmax_rel)
    ax3.set_title(f"{title_prefix}: Rel. diff (NN−Phys)/Phys")
    ax3.set_xlabel("phase")
    ax3.set_ylabel("energy channel")
    fig.colorbar(m3, cax=cax3, label="fraction")

    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved → {outfile}")

def plot_lightcurve_and_spectrum(map_phys: np.ndarray,
                                 map_nn: np.ndarray,
                                 title_prefix: str,
                                 outfile: Path,
                                 cycles: int = 2) -> None:
    """
    Plot bolometric lightcurves (sum over energy) and spectrograms (sum over phase).
    map_* are shape (nphase, nchan).
    """
    # Lightcurves
    lc_phys = map_phys.sum(axis=1)  # (nphase,)
    lc_nn   = map_nn.sum(axis=1)

    # Spectra
    spec_phys = map_phys.sum(axis=0)  # (nchan,)
    spec_nn   = map_nn.sum(axis=0)

    # Phase axis for multiple cycles
    nphase = map_phys.shape[0]
    base_phase = (np.arange(nphase) + 0.5) / nphase
    phase_axis = np.concatenate([base_phase + k for k in range(cycles)])
    lc_phys_plot = np.tile(lc_phys, cycles)
    lc_nn_plot   = np.tile(lc_nn, cycles)

    energy_axis = np.arange(map_phys.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=180)
    ax_lc, ax_sp = axes

    ax_lc.plot(phase_axis, lc_phys_plot, label="Physical", color="tab:blue", lw=2.0)
    ax_lc.plot(phase_axis, lc_nn_plot,   label="Neural Net", color="tab:orange", lw=2.0, ls="--")
    ax_lc.set_title(f"{title_prefix}: Bolometric lightcurve")
    ax_lc.set_xlabel("phase")
    ax_lc.set_ylabel("sum over energy")
    ax_lc.set_xlim(phase_axis.min(), phase_axis.max())
    ax_lc.legend(frameon=False)

    ax_sp.plot(energy_axis, spec_phys, label="Physical", color="tab:blue", lw=2.0)
    ax_sp.plot(energy_axis, spec_nn,   label="Neural Net", color="tab:orange", lw=2.0, ls="--")
    ax_sp.set_title(f"{title_prefix}: Spectrogram (sum over phase)")
    ax_sp.set_xlabel("energy channel")
    ax_sp.set_ylabel("sum over phase")
    ax_sp.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved → {outfile}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading photon counts:")
    print(f"  Physical: {COUNTS_PHYSICAL_FILE}")
    print(f"  NN      : {COUNTS_NN_FILE}")
    print(f"Loading MV211 flux (idistrib):")
    print(f"  Physical: {FLUX_PHYSICAL_FILE}")
    print(f"  NN      : {FLUX_NN_FILE}")

    # Load counts
    counts_phys = load_fortran_phase_major(COUNTS_PHYSICAL_FILE, NPHASE_MODEL, REAL_NCHAN_COUNTS)
    counts_nn   = load_fortran_phase_major(COUNTS_NN_FILE,       NPHASE_MODEL, REAL_NCHAN_COUNTS)

    print(f"\nCounts (nphase={NPHASE_MODEL}, nchan={REAL_NCHAN_COUNTS})")
    print(f"  Physical sum/min/max : {counts_phys.sum():.4e} / {counts_phys.min():.4e} / {counts_phys.max():.4e}")
    print(f"  NN sum/min/max       : {counts_nn.sum():.4e} / {counts_nn.min():.4e} / {counts_nn.max():.4e}")

    # Load fluxes (idistrib)
    flux_phys = load_fortran_phase_major(FLUX_PHYSICAL_FILE, NPHASE_MODEL, REAL_NCHAN_FLUX)
    flux_nn   = load_fortran_phase_major(FLUX_NN_FILE,       NPHASE_MODEL, REAL_NCHAN_FLUX)

    print(f"\nFlux (idistrib) (nphase={NPHASE_MODEL}, nchan={REAL_NCHAN_FLUX})")
    print(f"  Physical sum/min/max : {flux_phys.sum():.4e} / {flux_phys.min():.4e} / {flux_phys.max():.4e}")
    print(f"  NN sum/min/max       : {flux_nn.sum():.4e} / {flux_nn.min():.4e} / {flux_nn.max():.4e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2D maps + relative difference — Counts
    ep_counts_phys = to_energy_phase(counts_phys)
    ep_counts_nn   = to_energy_phase(counts_nn)
    plot_2d_comparison(
        ep_counts_phys, ep_counts_nn,
        title_prefix="J0030 Counts",
        outfile=OUTPUT_DIR / "j0030_counts_2d_comparison.png",
    )

    # LC + spectrum — Counts
    plot_lightcurve_and_spectrum(
        map_phys=counts_phys,
        map_nn=counts_nn,
        title_prefix="J0030 Counts",
        outfile=OUTPUT_DIR / "j0030_counts_lightcurve_spectrum.png",
        cycles=2,
    )

    # 2D maps + relative difference — Fluxes (idistrib)
    ep_flux_phys = to_energy_phase(flux_phys)
    ep_flux_nn   = to_energy_phase(flux_nn)
    plot_2d_comparison(
        ep_flux_phys, ep_flux_nn,
        title_prefix="J0030 Flux (idistrib)",
        outfile=OUTPUT_DIR / "j0030_flux_2d_comparison.png",
    )

    # LC + spectrum — Fluxes
    plot_lightcurve_and_spectrum(
        map_phys=flux_phys,
        map_nn=flux_nn,
        title_prefix="J0030 Flux (idistrib)",
        outfile=OUTPUT_DIR / "j0030_flux_lightcurve_spectrum.png",
        cycles=2,
    )

    print("\nDone.")

if __name__ == "__main__":
    main()