"""
plot_phase_energy_map.py

Reads a mcmc_vac_*.dat file and plots the 2D energy-phase map
(Idistrib) for two rows: row 0 (ia=0, original) and row N (ia=N,
phase-rotated by N/32 * 2pi), so the phase shift is visible.

File format (latest version):
  - 18 parameter values  (prms1art)
  -  1 loglik value
  - 32 * Nchan Idistrib values  (row-major: ibin=1..32, iE=1..Nchan)
  Total tokens per row: 19 + 32*Nchan
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Constants (must match the Fortran config used to produce the file)
# ---------------------------------------------------------------------------
NPRM     = 18    # number of parameter columns
NLOGLIK  = 1     # loglik column
NPHASE   = 32    # number of phase bins in Idistrib
NCHAN    = 1000  # number of intrinsic energy channels (nchan_cfg)
CHAN_SEP = 5.0   # channel separation in eV (chan_sep parameter in Fortran)
LOWER    = 1     # lower channel index (lower_cfg, 1-based)

# Fixed file paths for testcase comparison
TESTCASE_DIR      = Path(
    "/Users/jraynau1/Workspace/CLionProjects/ray_tracing_training_nn"
    "/output_data/j0740/mcmc_samples"
)
TESTCASE_OG_FILE  = TESTCASE_DIR / "mcmc_vac_100000_og.dat"   # "rotated"
TESTCASE_NEW_FILE = TESTCASE_DIR / "mcmc_vac_100000.dat"       # "non-rotated"
TESTCASE_OG_ROW   = 7   # 0-based index of the 7th row (1-based: row 8 -- wait, note below)
TESTCASE_NEW_ROW  = 0   # first row

# NOTE: TESTCASE_OG_ROW = 7 means 0-based index 7 = 8th row (1-based).
#       If you want the 7th row (1-based), set TESTCASE_OG_ROW = 6.

# Parameter names for printing
PARAM_NAMES = [
    "xD       ", "yD       ", "zD       ",
    "aD       ", "phiD     ",
    "xQ       ", "yQ       ", "zQ       ",
    "aQ       ", "phiQ     ",
    "BQ       ",
    "mass     ", "radius   ", "obsangle ",
    "T1       ",
    "distance ", "NH       ",
    "T2       ",
]

# Derived
N_HEADER = NPRM + NLOGLIK   # tokens before Idistrib
N_MAP    = NPHASE * NCHAN   # tokens in the map block
N_TOT    = N_HEADER + N_MAP # total tokens per row


# ---------------------------------------------------------------------------
# Energy axis reconstruction
# ---------------------------------------------------------------------------
def make_energy_axis(nchan: int, chan_sep_eV: float, lower: int) -> np.ndarray:
    """
    Reconstruct the central energy of each intrinsic bin in keV.
    """
    energies_eV = (np.arange(nchan) + lower) * chan_sep_eV
    return energies_eV / 1000.0  # eV -> keV


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------
def parse_row(filepath: Path, row_index: int, nchan: int = NCHAN) -> dict:
    """
    Parse a single row (0-based index) from a mcmc_vac file.

    Returns a dict with:
        'prms'     : (18,)        parameter array
        'loglik'   : float
        'idistrib' : (32, nchan)  energy-phase map
    """
    n_header = NPRM + NLOGLIK
    n_map    = NPHASE * nchan

    with open(filepath, "r") as fh:
        for i, line in enumerate(fh):
            if i == row_index:
                tokens = line.split()
                break
        else:
            raise IndexError(f"Row {row_index} not found in {filepath}")

    tokens = np.array(tokens, dtype=np.float64)

    if len(tokens) != n_header + n_map:
        raise ValueError(
            f"Row {row_index}: expected {n_header + n_map} tokens, "
            f"got {len(tokens)}. Check NCHAN ({nchan}) and NPHASE ({NPHASE})."
        )

    prms     = tokens[:NPRM]
    loglik   = tokens[NPRM]
    map_flat = tokens[n_header:]

    # Fortran writes ((Idistrib(ibin,iE), iE=1,Nchan), ibin=1,32)
    # -> shape (nphase, nchan) after reshape
    idistrib = map_flat.reshape(NPHASE, nchan)

    return {"prms": prms, "loglik": loglik, "idistrib": idistrib}


# ---------------------------------------------------------------------------
# Parameter printing
# ---------------------------------------------------------------------------
def print_parameters(row: dict, label: str, filepath: Path,
                     row_index: int) -> None:
    """
    Pretty-print the 18 parameters from a parsed row.
    """
    print(f"\n  {'='*55}")
    print(f"  Parameters for: {label}")
    print(f"  File   : {filepath.name}")
    print(f"  Row    : {row_index}  (0-based)  =  row {row_index+1}  (1-based)")
    print(f"  loglik : {row['loglik']:.6f}")
    print(f"  {'-'*55}")
    print(f"  {'#':>3}  {'Name':<12}  {'Value':>22}")
    print(f"  {'-'*55}")
    for i, (name, val) in enumerate(zip(PARAM_NAMES, row['prms'])):
        print(f"  {i+1:>3}  {name:<12}  {val:>22.15E}")
    print(f"  {'='*55}")


# ---------------------------------------------------------------------------
# 2D energy-phase map plot
# ---------------------------------------------------------------------------
def plot_two_maps(
        map_orig:              np.ndarray,
        map_rot:               np.ndarray,
        row_orig:              int,
        row_rot:               int,
        energy_axis:           np.ndarray,
        phase_shift:           float,
        title_orig:            str | None = None,
        title_rot:             str | None = None,
        suptitle:              str | None = None,
        output_path:           Path | None = None,
        show:                  bool = True,
        dynamic_range_decades: float = 4.0,
):
    """
    Side-by-side plot of two energy-phase maps with log colour scale.

    title_orig, title_rot : override panel titles (optional)
    suptitle              : override figure suptitle (optional)
    """
    phase_bins = np.arange(NPHASE)
    phase_axis = (phase_bins + 0.5) / NPHASE  # bin centres in [0, 1]

    combined = np.concatenate([map_orig.ravel(), map_rot.ravel()])
    nonzero  = combined[combined > 0]
    if len(nonzero) == 0:
        raise ValueError("All map values are zero — nothing to plot.")

    vmax = np.percentile(nonzero, 99)
    vmin = vmax / (10 ** 7)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    print(f"  vmax = {vmax:.3e},  vmin = {vmin:.3e}  "
          f"({dynamic_range_decades:.1f} decades)")
    print(f"  Full data range: {nonzero.min():.3e} – {nonzero.max():.3e}")
    print(f"  Fraction of cells above vmin: "
          f"orig={np.mean(map_orig >= vmin):.2%}, "
          f"rot={np.mean(map_rot  >= vmin):.2%}")

    # Default titles
    if title_orig is None:
        title_orig = f"Original  (row {row_orig},  ia=0,  Δφ = 0)"
    if title_rot is None:
        title_rot = (f"Rotated   (row {row_rot},  ia={row_rot},  "
                     f"Δφ = {phase_shift:.3f} cycles)")
    if suptitle is None:
        suptitle = (f"Energy–phase map: Idistrib  "
                    f"(log scale, {dynamic_range_decades:.0f}-decade range)")

    fig = plt.figure(figsize=(14, 5), dpi=150)
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.08)

    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    cax  = fig.add_subplot(gs[2])

    titles = [title_orig, title_rot]
    maps   = [map_orig, map_rot]

    mesh = None
    for ax, title, m in zip(axes, titles, maps):
        m_masked = np.ma.masked_where(m < vmin, m)
        mesh = ax.pcolormesh(
            phase_axis,
            energy_axis,
            m_masked.T,
            shading = "auto",
            cmap    = "magma",
            norm    = norm,
        )
        ax.set_facecolor("black")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Pulse phase (cycles)")
        ax.set_xlim(0, 1)
        ax.set_ylim(energy_axis.min(), energy_axis.max())

    axes[0].set_ylabel("Intrinsic energy (keV)")
    axes[1].set_yticklabels([])

    fig.colorbar(
        mesh, cax=cax,
        label=r"$I_\nu$ (erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ bin$^{-1}$)",
    )
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Testcase comparison: summed over phase bins  (spectrum comparison)
# ---------------------------------------------------------------------------
def plot_summed_over_phase(
        map_og:      np.ndarray,
        map_new:     np.ndarray,
        energy_axis: np.ndarray,
        phase_roll:  int = 0,
        output_path: Path | None = None,
        show:        bool = True,
):
    """
    Sum Idistrib over all 32 phase bins → spectrum I(E).
    phase_roll: number of bins to roll map_og to undo the ia rotation.
    """
    map_og_rolled = np.roll(map_og, -phase_roll, axis=0)

    spectrum_og  = map_og_rolled.sum(axis=0)
    spectrum_new = map_new.sum(axis=0)

    # Channel indices: 1-based to match Fortran (lower=1 by default)
    nchan       = spectrum_og.shape[0]
    chan_axis   = np.arange(LOWER, LOWER + nchan)   # channel numbers 1..Nchan

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    ax.plot(chan_axis, spectrum_og,  color="C1", lw=1.5,
            label=f"Rotated  ({TESTCASE_OG_FILE.name},  row {TESTCASE_OG_ROW+1}, "
                  f"rolled by {phase_roll} bins)")
    ax.plot(chan_axis, spectrum_new, color="C0", lw=1.5, ls="--",
            label=f"Non-rotated  ({TESTCASE_NEW_FILE.name},  row {TESTCASE_NEW_ROW+1})")

    ax.set_xlabel(f"Energy channel index  (1-based, Nchan={nchan})")
    ax.set_ylabel(
        r"$\sum_{\rm phase}\,I_\nu$  "
        r"(erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ bin$^{-1}$)"
    )
    ax.set_title("Phase-summed spectrum: testcase comparison")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.set_xlim(chan_axis.min(), chan_axis.max())
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Testcase comparison: summed over energy bins  (light-curve comparison)
# ---------------------------------------------------------------------------
def plot_summed_over_energy(
        map_og:      np.ndarray,
        map_new:     np.ndarray,
        phase_roll:  int = 0,
        output_path: Path | None = None,
        show:        bool = True,
):
    """
    Sum Idistrib over all Nchan energy bins → light curve I(phase).
    phase_roll: number of bins to roll map_og to undo the ia rotation.
    """
    map_og_rolled = np.roll(map_og, -phase_roll, axis=0)

    lightcurve_og  = map_og_rolled.sum(axis=1)
    lightcurve_new = map_new.sum(axis=1)

    lc_og_raw = map_og.sum(axis=1)

    phase_bins = np.arange(NPHASE)
    phase_axis = (phase_bins + 0.5) / NPHASE

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left panel: as stored in file (with phase shift)
    axes[0].plot(phase_axis, lc_og_raw,      color="C1", lw=1.5, marker="o",
                 ms=4, label=f"Rotated  (row {TESTCASE_OG_ROW+1}, as stored)")
    axes[0].plot(phase_axis, lightcurve_new, color="C0", lw=1.5, marker="s",
                 ms=4, ls="--",
                 label=f"Non-rotated  (row {TESTCASE_NEW_ROW+1})")
    axes[0].set_title("As stored in file (phase-shifted)")
    axes[0].set_xlabel("Pulse phase (cycles)")
    axes[0].set_ylabel(r"$\sum_{E}\,I_\nu$")
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8)

    # Right panel: after rolling to undo ia rotation
    axes[1].plot(phase_axis, lightcurve_og,  color="C1", lw=1.5, marker="o",
                 ms=4,
                 label=f"Rotated  (row {TESTCASE_OG_ROW+1}, "
                       f"rolled by {phase_roll} bins)")
    axes[1].plot(phase_axis, lightcurve_new, color="C0", lw=1.5, marker="s",
                 ms=4, ls="--",
                 label=f"Non-rotated  (row {TESTCASE_NEW_ROW+1})")
    axes[1].set_title(
        f"After rolling og by {phase_roll} bins  "
        f"(Δφ = {phase_roll}/32 = {phase_roll/32:.3f} cycles)"
    )
    axes[1].set_xlabel("Pulse phase (cycles)")
    axes[1].set_ylabel(r"$\sum_{E}\,I_\nu$")
    axes[1].set_xlim(0, 1)
    axes[1].legend(fontsize=8)

    plt.suptitle("Energy-summed light curve: testcase comparison", fontsize=11)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot energy-phase maps from a mcmc_vac_*.dat file."
    )
    parser.add_argument(
        "filepath",
        type=Path,
        nargs="?",
        default=Path(
            "/Users/jraynau1/Workspace/CLionProjects/ray_tracing_training_nn"
            "/output_data/j0740/mcmc_samples/mcmc_vac_100000.dat"
        ),
        help="Path to mcmc_vac_*.dat file.",
    )
    parser.add_argument(
        "--row-orig", type=int, default=0,
        help="0-based row index for the original map (default: 0, ia=0).",
    )
    parser.add_argument(
        "--row-rot", type=int, nargs="+", default=[1, 3, 7, 11, 15, 23, 31],
        help="One or more 0-based row indices for rotated maps.",
    )
    parser.add_argument(
        "--nchan", type=int, default=NCHAN,
        help=f"Number of intrinsic energy channels (default: {NCHAN}).",
    )
    parser.add_argument(
        "--chan-sep", type=float, default=CHAN_SEP,
        help=f"Channel separation in eV (default: {CHAN_SEP}).",
    )
    parser.add_argument(
        "--lower", type=int, default=LOWER,
        help=f"Lower channel index, 1-based (default: {LOWER}).",
    )
    parser.add_argument(
        "--dynamic-range", type=float, default=4.0,
        help=(
            "Number of decades below vmax to display (default: 4.0). "
            "Increase to reveal fainter structure; decrease to suppress noise."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "/Users/jraynau1/Workspace/data_analysis_visualization"
            "/training_nn/plots"
        ),
        help="Directory to save figures (one PNG per rotated row).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display interactive plots (useful for batch runs).",
    )
    parser.add_argument(
        "--testcase", action="store_true",
        help=(
            "Also plot testcase comparison: 2D maps, phase-summed spectrum and "
            "energy-summed light curve for og vs new files."
        ),
    )
    args = parser.parse_args()

    energy_axis = make_energy_axis(args.nchan, args.chan_sep, args.lower)

    # -----------------------------------------------------------------------
    # Standard 2D phase-energy map plots
    # -----------------------------------------------------------------------
    print(f"Reading {args.filepath} ...")
    print(f"  Expected tokens per row : {N_HEADER + NPHASE * args.nchan}")

    row_orig = parse_row(args.filepath, args.row_orig, nchan=args.nchan)

    print(f"  Row {args.row_orig} (original): loglik = {row_orig['loglik']:.6f}")
    print(f"  Idistrib shape: {row_orig['idistrib'].shape}  "
          f"(nphase={NPHASE}, nchan={args.nchan})")

    for row_idx in args.row_rot:
        print(f"\n--- Comparing row {args.row_orig} vs row {row_idx} ---")
        row_rot     = parse_row(args.filepath, row_idx, nchan=args.nchan)
        phase_shift = row_idx / 32.0
        print(f"  Row {row_idx} (rotated):  loglik = {row_rot['loglik']:.6f}, "
              f"Δφ = {phase_shift:.4f} cycles  ({row_idx}/32 × 2π)")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = (args.output_dir /
                    f"phase_map_orig{args.row_orig}_rot{row_idx}"
                    f"_dr{args.dynamic_range:.0f}.png")

        plot_two_maps(
            map_orig              = row_orig["idistrib"],
            map_rot               = row_rot["idistrib"],
            row_orig              = args.row_orig,
            row_rot               = row_idx,
            energy_axis           = energy_axis,
            phase_shift           = phase_shift,
            output_path           = out_path,
            show                  = not args.no_show,
            dynamic_range_decades = args.dynamic_range,
        )

    # -----------------------------------------------------------------------
    # Testcase comparison plots
    # -----------------------------------------------------------------------
    if args.testcase:
        print(f"\n{'='*60}")
        print(f"TESTCASE COMPARISON")
        print(f"{'='*60}")
        print(f"  OG file  (rotated)     : {TESTCASE_OG_FILE},  row {TESTCASE_OG_ROW} "
              f"(0-based) = row {TESTCASE_OG_ROW+1} (1-based)")
        print(f"  New file (non-rotated) : {TESTCASE_NEW_FILE}, row {TESTCASE_NEW_ROW} "
              f"(0-based) = row {TESTCASE_NEW_ROW+1} (1-based)")

        if not TESTCASE_OG_FILE.exists():
            print(f"  WARNING: OG file not found: {TESTCASE_OG_FILE}")
            return
        if not TESTCASE_NEW_FILE.exists():
            print(f"  WARNING: New file not found: {TESTCASE_NEW_FILE}")
            return

        row_og  = parse_row(TESTCASE_OG_FILE,  TESTCASE_OG_ROW,  nchan=args.nchan)
        row_new = parse_row(TESTCASE_NEW_FILE, TESTCASE_NEW_ROW, nchan=args.nchan)

        # Print parameters for both rows
        print_parameters(
            row       = row_og,
            label     = "Rotated (OG file)",
            filepath  = TESTCASE_OG_FILE,
            row_index = TESTCASE_OG_ROW,
        )
        print_parameters(
            row       = row_new,
            label     = "Non-rotated (new file)",
            filepath  = TESTCASE_NEW_FILE,
            row_index = TESTCASE_NEW_ROW,
        )

        # Side-by-side parameter diff
        print(f"\n  {'='*55}")
        print(f"  Parameter differences  (OG - New)")
        print(f"  {'-'*55}")
        print(f"  {'#':>3}  {'Name':<12}  {'OG':>22}  {'New':>22}  {'Diff':>22}")
        print(f"  {'-'*55}")
        for i, (name, v_og, v_new) in enumerate(
                zip(PARAM_NAMES, row_og['prms'], row_new['prms'])):
            diff = v_og - v_new
            flag = "  ← DIFFERS" if abs(diff) > 1e-12 else ""
            print(f"  {i+1:>3}  {name:<12}  {v_og:>22.15E}  "
                  f"{v_new:>22.15E}  {diff:>22.15E}{flag}")
        print(f"  {'='*55}")

        # ---------------------------------------------------------------
        # Numerical comparison of Idistrib arrays
        # ---------------------------------------------------------------
        diff_map = row_og["idistrib"] - row_new["idistrib"]
        abs_diff = np.abs(diff_map)
        rel_diff = abs_diff / (np.abs(row_new["idistrib"]) + 1e-300)

        print(f"\n  {'='*55}")
        print(f"  Idistrib numerical comparison  (OG row {TESTCASE_OG_ROW+1}"
              f" vs New row {TESTCASE_NEW_ROW+1})")
        print(f"  {'-'*55}")
        print(f"  Shape                       : {row_og['idistrib'].shape}")
        print(f"  OG  sum(Idistrib)           : {row_og['idistrib'].sum():.6e}")
        print(f"  New sum(Idistrib)           : {row_new['idistrib'].sum():.6e}")
        print(f"  Max absolute difference     : {abs_diff.max():.6e}")
        print(f"  Mean absolute difference    : {abs_diff.mean():.6e}")
        print(f"  Max relative difference     : {rel_diff.max():.6e}")
        print(f"  Fraction of cells differing : "
              f"{np.mean(abs_diff > 0):.2%}")
        print(f"  Maps identical (rtol=1e-10) : "
              f"{np.allclose(row_og['idistrib'], row_new['idistrib'], rtol=1e-10, atol=0)}")
        print(f"  Maps identical (rtol=1e-6)  : "
              f"{np.allclose(row_og['idistrib'], row_new['idistrib'], rtol=1e-6,  atol=0)}")
        print(f"  {'='*55}")

        # Phase roll to undo ia rotation stored in og row
        phase_roll = TESTCASE_OG_ROW
        print(f"\n  Phase roll to undo og ia rotation : {phase_roll} bins "
              f"= {phase_roll}/32 = {phase_roll/32:.4f} cycles "
              f"= {phase_roll/32*360:.2f} deg")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------
        # Figure 0: 2D energy-phase maps side by side (as stored)
        # ---------------------------------------------------------------
        print(f"\n--- Testcase 2D map (as stored, no roll) ---")
        plot_two_maps(
            map_orig    = row_new["idistrib"],
            map_rot     = row_og["idistrib"],
            row_orig    = TESTCASE_NEW_ROW,
            row_rot     = TESTCASE_OG_ROW,
            energy_axis = energy_axis,
            phase_shift = phase_roll / 32.0,
            title_orig  = (f"Non-rotated  ({TESTCASE_NEW_FILE.name}, "
                           f"row {TESTCASE_NEW_ROW+1})"),
            title_rot   = (f"Rotated  ({TESTCASE_OG_FILE.name}, "
                           f"row {TESTCASE_OG_ROW+1},  "
                           f"Δφ = {phase_roll}/32 = {phase_roll/32:.3f} cycles)"),
            suptitle    = ("Testcase 2D energy–phase map: as stored  "
                           "(log scale, 7-decade range)"),
            output_path = args.output_dir / "testcase_2d_map_as_stored.png",
            show        = not args.no_show,
            dynamic_range_decades = args.dynamic_range,
        )

        # ---------------------------------------------------------------
        # Figure 0b: 2D energy-phase maps side by side (og rolled)
        # ---------------------------------------------------------------
        print(f"\n--- Testcase 2D map (og rolled by {phase_roll} bins) ---")
        map_og_rolled = np.roll(row_og["idistrib"], -phase_roll, axis=0)
        plot_two_maps(
            map_orig    = row_new["idistrib"],
            map_rot     = map_og_rolled,
            row_orig    = TESTCASE_NEW_ROW,
            row_rot     = TESTCASE_OG_ROW,
            energy_axis = energy_axis,
            phase_shift = 0.0,
            title_orig  = (f"Non-rotated  ({TESTCASE_NEW_FILE.name}, "
                           f"row {TESTCASE_NEW_ROW+1})"),
            title_rot   = (f"Rotated  ({TESTCASE_OG_FILE.name}, "
                           f"row {TESTCASE_OG_ROW+1},  "
                           f"rolled by {phase_roll} bins)"),
            suptitle    = (f"Testcase 2D energy–phase map: og rolled by "
                           f"{phase_roll} bins  (log scale, 7-decade range)"),
            output_path = args.output_dir / "testcase_2d_map_rolled.png",
            show        = not args.no_show,
            dynamic_range_decades = args.dynamic_range,
        )

        # ---------------------------------------------------------------
        # Figure 1: summed over phase bins → spectrum
        # ---------------------------------------------------------------
        plot_summed_over_phase(
            map_og      = row_og["idistrib"],
            map_new     = row_new["idistrib"],
            energy_axis = energy_axis,
            phase_roll  = phase_roll,
            output_path = args.output_dir / "testcase_summed_over_phase.png",
            show        = not args.no_show,
        )

        # ---------------------------------------------------------------
        # Figure 2: summed over energy bins → light curve
        # ---------------------------------------------------------------
        plot_summed_over_energy(
            map_og      = row_og["idistrib"],
            map_new     = row_new["idistrib"],
            phase_roll  = phase_roll,
            output_path = args.output_dir / "testcase_summed_over_energy.png",
            show        = not args.no_show,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()