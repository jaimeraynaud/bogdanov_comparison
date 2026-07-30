"""
compute_j0030_mdnse.py

Reads the NICER observed 1-D light curve from data/nicer_profile.dat and
computes the Median Normalised Squared Error (MdNSE) uncertainty metric.

nicer_profile.dat column layout (0-based):
    col 0 : rotational phase
    col 1 : observed photon counts  (raw, background not yet subtracted)

Definitions
-----------
    raw_i        : observed counts in phase bin i  (col 1)
    bkg_sub_i    : background-subtracted counts    = raw_i − BACKGROUND
    uncertainty_i: Poisson uncertainty             = sqrt(raw_i + BACKGROUND)

    MdNSE = sum(uncertainty_i²) / median(bkg_sub)²
          = sum(raw_i + BACKGROUND) / median(bkg_sub)²

The MdNSE gives a dimensionless measure of how large the total squared
uncertainty is relative to the typical (median) signal level squared.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NICER_FILE = Path(__file__).parent / "data" / "nicer_profile.dat"

OUTPUT_DIR = Path(
    "/Users/jraynau1/Workspace/data_analysis_visualization/training_nn/plots"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKGROUND = 13500    # constant background counts per phase bin


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------
def load_nicer_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load nicer_profile.dat (two columns: phase, raw counts).

    Returns
    -------
    phase     : (n,)  — phase values
    raw_counts: (n,)  — raw observed photon counts (background NOT subtracted)
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected a 2-column file, got shape {data.shape}"
        )
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Uncertainty and MdNSE
# ---------------------------------------------------------------------------
def compute_uncertainty(raw_counts: np.ndarray,
                        background: float = BACKGROUND) -> np.ndarray:
    """
    Per-bin Poisson uncertainty = sqrt(raw_counts + background).

    Parameters
    ----------
    raw_counts : (n,)  — raw observed counts per phase bin
    background : constant background level

    Returns
    -------
    uncertainty : (n,)
    """
    return np.sqrt(raw_counts + background)


def compute_mdnse(uncertainty: np.ndarray,
                  bkg_sub: np.ndarray) -> float:
    """
    Median Normalised Squared Error (MdNSE).

        MdNSE = sum(uncertainty_i²) / median(bkg_sub)²

    Parameters
    ----------
    uncertainty : (n,)  — per-bin uncertainty = sqrt(raw + background)
    bkg_sub     : (n,)  — background-subtracted counts = raw − background

    Returns
    -------
    MdNSE : float
    """
    numerator   = np.sum(uncertainty ** 2)          # = sum(raw + background)
    denominator = np.median(bkg_sub) ** 2
    if denominator == 0:
        raise ValueError(
            "Median of background-subtracted counts is zero — "
            "cannot normalise MdNSE."
        )
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_profile_with_uncertainties(phase:       np.ndarray,
                                    bkg_sub:     np.ndarray,
                                    uncertainty: np.ndarray,
                                    mdnse:       float,
                                    output_dir:  Path,
                                    cycles:      int = 2,
                                    show:        bool = True) -> None:
    """
    Plot the background-subtracted light curve with per-bin uncertainty
    error bars and annotate with the MdNSE value.

    Parameters
    ----------
    phase       : (n,)  — phase values
    bkg_sub     : (n,)  — background-subtracted counts
    uncertainty : (n,)  — per-bin uncertainty
    mdnse       : float — computed MdNSE
    cycles      : how many pulse cycles to show
    """
    # Tile for multiple cycles
    phase_ax    = np.concatenate([phase + k for k in range(cycles)])
    bkg_sub_ax  = np.tile(bkg_sub,     cycles)
    uncert_ax   = np.tile(uncertainty, cycles)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=180,
                             gridspec_kw={"height_ratios": [3, 1],
                                          "hspace": 0.08})

    # --- Top panel: light curve + error bars ---
    ax_top = axes[0]
    ax_top.errorbar(
        phase_ax, bkg_sub_ax, yerr=uncert_ax,
        fmt="o", ms=3.5, lw=0, elinewidth=1.2,
        color="tab:blue", ecolor="tab:blue", alpha=0.8,
        label="Observed  (bkg subtracted)",
    )
    ax_top.axhline(np.median(bkg_sub), color="tab:orange", ls="--", lw=1.4,
                   label=f"Median = {np.median(bkg_sub):.1f} counts")
    ax_top.set_ylabel("counts  (background subtracted)")
    ax_top.set_xlim(phase_ax.min(), phase_ax.max())
    ax_top.legend(frameon=False, fontsize=9)
    ax_top.set_title(
        f"J0030 NICER light curve  —  MdNSE = {mdnse:.4f}",
        fontsize=11,
    )
    ax_top.tick_params(labelbottom=False)

    # --- Bottom panel: per-bin uncertainty ---
    ax_bot = axes[1]
    ax_bot.bar(phase_ax, uncert_ax,
               width=(phase_ax[1] - phase_ax[0]) * 0.85,
               color="tab:gray", alpha=0.7,
               label=r"$\sigma_i = \sqrt{n_i + B}$")
    ax_bot.set_xlabel("phase")
    ax_bot.set_ylabel(r"$\sigma_i$  (counts)")
    ax_bot.set_xlim(phase_ax.min(), phase_ax.max())
    ax_bot.legend(frameon=False, fontsize=9)

    fig.tight_layout()

    out = output_dir / "j0030_nicer_profile_uncertainty.png"
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
    print(f"Loading NICER profile : {NICER_FILE}")
    print(f"Constant background   : {BACKGROUND:,} counts\n")

    # --- Load ---
    phase, raw_counts = load_nicer_profile(NICER_FILE)
    print(f"  Phase bins           : {len(phase)}")
    print(f"  Raw counts  min/max  : {raw_counts.min():.1f} / {raw_counts.max():.1f}")

    # --- Background-subtracted light curve ---
    bkg_sub = raw_counts - BACKGROUND
    print(f"\n  Bkg-subtracted min/max : {bkg_sub.min():.1f} / {bkg_sub.max():.1f}")
    print(f"  Bkg-subtracted median  : {np.median(bkg_sub):.1f}")

    # --- Per-bin uncertainty ---
    uncertainty = compute_uncertainty(raw_counts, background=BACKGROUND)
    print(f"\n  Uncertainty  min/max   : {uncertainty.min():.3f} / {uncertainty.max():.3f}")
    print(f"  sum(uncertainty²)      : {np.sum(uncertainty**2):.4e}")

    # --- MdNSE ---
    mdnse = compute_mdnse(uncertainty, bkg_sub)
    print(f"\n  MdNSE = sum(σ²) / median(bkg_sub)²")
    print(f"        = {np.sum(uncertainty**2):.4e} / {np.median(bkg_sub)**2:.4e}")
    print(f"        = {mdnse:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_profile_with_uncertainties(
        phase       = phase,
        bkg_sub     = bkg_sub,
        uncertainty = uncertainty,
        mdnse       = mdnse,
        output_dir  = OUTPUT_DIR,
        cycles      = 2,
        show        = True,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()