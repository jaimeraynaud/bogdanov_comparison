#!/usr/bin/env python3
"""Compare individual combined spots vs the combined spot output."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set the target directory based on your request
DEFAULT_NAS_DIR = Path("/Users/jraynau1/Workspace/data_analysis_visualization/j0030/data/combined/bfmirroringbig")
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "plots"


def load_nas_spot_counts(path: Path) -> np.ndarray:
    """Load a NAS spot count matrix from disk."""
    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D matrix for NAS spot counts, got shape {data.shape}")
    return data


def fractional_circular_shift(values: np.ndarray, shift_bins: float, axis: int = -1) -> np.ndarray:
    """Circularly shift ``values`` by a (possibly fractional) number of bins along ``axis``."""
    n = values.shape[axis]
    freqs = np.fft.rfft(values, axis=axis)
    k = np.arange(freqs.shape[axis])
    phase = np.exp(2j * np.pi * k * shift_bins / n)
    shape = [1] * values.ndim
    shape[axis] = phase.size
    phase = phase.reshape(shape)
    return np.fft.irfft(freqs * phase, n=n, axis=axis)


def get_bolometric_curve(matrix: np.ndarray, phase_axis: int = 0, shift_bins: float = 21.25) -> np.ndarray:
    """Sum the 2D matrix over the energy axis and apply phase shift.

    Parameters
    ----------
    phase_axis : int
        Which axis of the matrix corresponds to phase (0 for rows, 1 for columns).
        The curve is produced by summing over the OTHER axis.
    """
    # Sum over the energy axis (the one that is NOT the phase axis)
    energy_axis = 1 - phase_axis
    curve = matrix.sum(axis=energy_axis)

    # Apply the same phase shift as the original template
    if shift_bins != 0.0:
        curve = fractional_circular_shift(curve, shift_bins)

    return curve


def plot_comparison(
        spot1_curve: np.ndarray,
        spot2_curve: np.ndarray,
        combined_curve: np.ndarray,
        output_dir: Path,
        cycles: float = 2.0,
        show: bool = True
):
    """Plot the explicitly summed spots versus the combined file output."""
    summed_curve = spot1_curve + spot2_curve
    phase_count = len(summed_curve)

    # Base phase [0, 1) treated as bin centers
    base_phase = (np.arange(phase_count) + 0.5) / phase_count

    # Repeat for multiple cycles
    if float(cycles).is_integer() and int(cycles) >= 2:
        n_cycles = int(round(cycles))
        phase_axis = np.concatenate([base_phase + k for k in range(n_cycles)])

        summed_plot = np.tile(summed_curve, n_cycles)
        combined_plot = np.tile(combined_curve, n_cycles)
        spot1_plot = np.tile(spot1_curve, n_cycles)
        spot2_plot = np.tile(spot2_curve, n_cycles)
    else:
        phase_axis = base_phase * cycles
        summed_plot = summed_curve
        combined_plot = combined_curve
        spot1_plot = spot1_curve
        spot2_plot = spot2_curve

    fig, (ax_main, ax_diff) = plt.subplots(
        2, 1, figsize=(9, 7), dpi=180,
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )

    # --- Top Plot: Lightcurves ---
    # Draw combined file with a thick line
    ax_main.plot(phase_axis, combined_plot, label="Combined File", color="tab:blue", lw=3.0)

    # Draw explicitly summed curve with a dashed line on top so you can see if they match perfectly
    ax_main.plot(phase_axis, summed_plot, label="Spot 1 + Spot 2 (Calculated)", color="tab:orange", lw=2.0, ls="--")

    # Draw individual spots as thin guides
    ax_main.plot(phase_axis, spot1_plot, label="Spot 1", color="tab:cyan", lw=1.0, ls=":")
    ax_main.plot(phase_axis, spot2_plot, label="Spot 2", color="tab:red", lw=1.0, ls=":")

    ax_main.set_title("Validation: Summed Spots vs Combined File")
    ax_main.set_ylabel("summed counts")
    ax_main.legend(frameon=False)

    # --- Bottom Plot: Residuals (Difference) ---
    difference = summed_plot - combined_plot
    ax_diff.plot(phase_axis, difference, color="black", lw=1.5)
    ax_diff.axhline(0, color="gray", lw=1.0, ls="--")
    ax_diff.set_ylabel("Difference\n(Summed - Combined)")
    ax_diff.set_xlabel("phase")
    ax_diff.set_xlim(phase_axis.min(), phase_axis.max())

    fig.tight_layout()

    # Save and show
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "spot_combination_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare spot1 + spot2 vs combined spot file.")
    parser.add_argument(
        "--nas-dir",
        type=Path,
        default=DEFAULT_NAS_DIR,
        help=f"Directory containing NAS spot count files (default: {DEFAULT_NAS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=2.0,
        help="Number of cycles to display on the phase axis (default: 2.0).",
    )
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=None,
        help="Expected number of phase bins (default: auto-detect from file shape).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Target directory: {args.nas_dir}")

    # Define file paths
    spot1_path = args.nas_dir / "spot1_test_data_counts.dat"
    spot2_path = args.nas_dir / "spot2_test_data_counts.dat"
    combined_path = args.nas_dir / "spot_combined_test_data_counts.dat"

    # Verify files exist
    for p in [spot1_path, spot2_path, combined_path]:
        if not p.exists():
            print(f"Error: Required file not found: {p}")
            return

    # Load data
    mat_spot1 = load_nas_spot_counts(spot1_path)
    mat_spot2 = load_nas_spot_counts(spot2_path)
    mat_combined = load_nas_spot_counts(combined_path)

    print(f"Loaded shapes: spot1={mat_spot1.shape}, spot2={mat_spot2.shape}, combined={mat_combined.shape}")

    # Sanity check: all matrices should have the same shape
    if not (mat_spot1.shape == mat_spot2.shape == mat_combined.shape):
        print("Warning: input matrices have different shapes!")

    # Auto-detect which axis is the phase axis.
    # If user provided --phase-bins, use that; otherwise, assume the smaller axis is phase.
    if args.phase_bins is not None:
        if args.phase_bins == mat_spot1.shape[0]:
            phase_axis = 0
        elif args.phase_bins == mat_spot1.shape[1]:
            phase_axis = 1
        else:
            raise ValueError(
                f"--phase-bins={args.phase_bins} not found in matrix shape {mat_spot1.shape}"
            )
    else:
        # Heuristic: phase axis is typically the smaller of the two dimensions
        # (e.g., 32 phase bins vs. 260 energy channels).
        phase_axis = 0 if mat_spot1.shape[0] < mat_spot1.shape[1] else 1

    n_phase_bins = mat_spot1.shape[phase_axis]
    print(f"Using phase_axis={phase_axis} -> {n_phase_bins} phase bins")

    # Convert to 1D bolometric curves (with the standard phase shift of 21.25)
    curve_spot1 = get_bolometric_curve(mat_spot1, phase_axis=phase_axis)
    curve_spot2 = get_bolometric_curve(mat_spot2, phase_axis=phase_axis)
    curve_combined = get_bolometric_curve(mat_combined, phase_axis=phase_axis)

    # Plot
    plot_comparison(
        spot1_curve=curve_spot1,
        spot2_curve=curve_spot2,
        combined_curve=curve_combined,
        output_dir=args.output_dir,
        cycles=args.cycles,
        show=not args.no_show
    )


if __name__ == "__main__":
    main()