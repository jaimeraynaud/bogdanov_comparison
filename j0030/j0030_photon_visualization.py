#!/usr/bin/env python3
"""Visualize the J0030 phase-channel model and NAS spot-count comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_FILE = Path(__file__).parent / "data" / "j0030_phase_channel_model.txt"
DEFAULT_NAS_DIR = Path(__file__).parent / "data" / "latest2"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "plots"


def load_phase_channel_table(path: Path) -> np.ndarray:
    """Load the phase-channel model table from disk."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            f"Expected a 2D table with at least 4 columns, got shape {data.shape}"
        )
    return data


def load_nas_spot_counts(path: Path) -> np.ndarray:
    """Load a NAS spot count matrix from disk."""
    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D matrix for NAS spot counts, got shape {data.shape}")
    return data


def infer_phase_energy_map(data: np.ndarray, value_column: int = 3):
    """
    Reshape the table into an energy x phase map.

    The J0030 file stores explicit energy and phase coordinates in the first
    two columns. We use those coordinates to reshape the selected value column
    instead of assuming the table is already a raw matrix.
    """
    if data.shape[1] <= value_column:
        raise ValueError(
            f"Requested value column {value_column} but table only has {data.shape[1]} columns"
        )

    energy_vals = np.unique(data[:, 0])
    phase_vals = np.unique(data[:, 1])

    # The file should contain one row per (energy, phase) pair.
    expected_rows = len(energy_vals) * len(phase_vals)
    if data.shape[0] != expected_rows:
        raise ValueError(
            f"Table shape does not match unique coordinate pairs: "
            f"{data.shape[0]} rows vs {len(energy_vals)} energies x {len(phase_vals)} phases"
        )

    # Sort to ensure the reshape is stable and predictable.
    order = np.lexsort((data[:, 1], data[:, 0]))
    sorted_data = data[order]

    phase_energy_map = sorted_data[:, value_column].reshape(len(energy_vals), len(phase_vals))

    return energy_vals, phase_vals, phase_energy_map


def plot_phase_energy_map(
    energy_vals: np.ndarray,
    phase_vals: np.ndarray,
    phase_energy_map: np.ndarray,
    title: str,
    output_path: Path | None = None,
    show: bool = True,
):
    """Plot the 2D phase-energy map."""
    # Treat the phase labels as bin centers if they are integer indices.
    if np.allclose(phase_vals, np.round(phase_vals)):
        phase_axis = (phase_vals + 0.5) / len(phase_vals)
        phase_label = "phase"
    else:
        phase_axis = phase_vals
        phase_label = "phase"

    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    mesh = ax.pcolormesh(
        phase_axis,
        energy_vals,
        phase_energy_map,
        shading="auto",
        cmap="magma",
    )
    fig.colorbar(mesh, ax=ax, label="counts")
    ax.set_title(title)
    ax.set_xlabel(phase_label)
    ax.set_ylabel("energy")
    ax.set_xlim(phase_axis.min(), phase_axis.max())
    ax.set_ylim(energy_vals.min(), energy_vals.max())
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved phase-energy map to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bolometric_light_curve(
    phase_vals: np.ndarray,
    phase_energy_map: np.ndarray,
    title: str,
    output_path: Path | None = None,
    show: bool = True,
    overlay_curves: list[tuple[str, np.ndarray, dict]] | None = None,
):
    """Plot the bolometric light curve by summing over energy bins.

    Parameters
    ----------
    overlay_curves:
        Optional list of comparison curves to draw on the same axes. Each entry
        is ``(label, curve, style_kwargs)`` and the curve must already be
        aligned to the phase axis.
    """
    bolometric = phase_energy_map.sum(axis=0)

    if np.allclose(phase_vals, np.round(phase_vals)):
        phase_axis = (phase_vals + 0.5) / len(phase_vals)
        phase_label = "phase"
    else:
        phase_axis = phase_vals
        phase_label = "phase"

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)
    ax.plot(phase_axis, bolometric, color="tab:blue", lw=2.2, label="model")

    if overlay_curves:
        for label, curve, style_kwargs in overlay_curves:
            ax.plot(phase_axis, curve, label=label, **style_kwargs)

    ax.set_title(title)
    ax.set_xlabel(phase_label)
    ax.set_ylabel("summed counts")
    ax.set_xlim(phase_axis.min(), phase_axis.max())
    ax.legend(frameon=False)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved bolometric light curve to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_phase_energy_and_bolometric(
    energy_vals: np.ndarray,
    phase_vals: np.ndarray,
    phase_energy_map: np.ndarray,
    title_prefix: str,
    output_dir: Path,
    stem: str,
    show: bool,
    overlay_curves: list[tuple[str, np.ndarray, dict]] | None = None,
    overlay_stem_suffix: str | None = None,
):
    """Save the phase-energy map and bolometric light curve for one data column."""
    map_path = output_dir / f"{stem}_phase_energy_map.png"
    lc_stem = stem if overlay_stem_suffix is None else f"{stem}_{overlay_stem_suffix}"
    lc_path = output_dir / f"{lc_stem}_bolometric_light_curve.png"

    plot_phase_energy_map(
        energy_vals=energy_vals,
        phase_vals=phase_vals,
        phase_energy_map=phase_energy_map,
        title=f"J0030 {title_prefix} Phase-Energy Map",
        output_path=map_path,
        show=False,
    )
    plot_bolometric_light_curve(
        phase_vals=phase_vals,
        phase_energy_map=phase_energy_map,
        title=f"J0030 {title_prefix} Bolometric Light Curve",
        output_path=lc_path,
        show=show,
        overlay_curves=overlay_curves,
    )


def load_nas_spot_curves(nas_dir: Path, phase_count: int) -> dict[str, np.ndarray]:
    """Load NAS spot count matrices and convert them to phase-bolometric curves."""
    spot_curves: dict[str, np.ndarray] = {}
    for spot_name in ("spot1", "spot2", "spot3"):
        path = nas_dir / f"{spot_name}_test_data_counts.dat"
        matrix = load_nas_spot_counts(path)

        if phase_count not in matrix.shape:
            raise ValueError(
                f"{path} has shape {matrix.shape}, which does not contain the expected "
                f"phase-bin count {phase_count}"
            )

        if matrix.shape[0] == phase_count:
            curve = matrix.sum(axis=1)
        elif matrix.shape[1] == phase_count:
            curve = matrix.sum(axis=0)
        else:
            raise ValueError(
                f"Could not infer phase axis for {path} with shape {matrix.shape}"
            )

        # Apply the same phase shift as the model (15 bins)
        curve = np.concatenate((curve[23:], curve[:23]))
        spot_curves[spot_name] = curve

    return spot_curves


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the J0030 phase-channel model as a 2D map and bolometric curve."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help=f"Input phase-channel model file (default: {DEFAULT_DATA_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--nas-dir",
        type=Path,
        default=DEFAULT_NAS_DIR,
        help=f"Directory containing NAS spot count files (default: {DEFAULT_NAS_DIR})",
    )
    parser.add_argument(
        "--value-columns",
        type=int,
        nargs="+",
        default=[4],
        help=(
            "Zero-based data columns to plot. Default: 2 3 "
            "(observed counts and best-fit model counts)."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_phase_channel_table(input_file)
    unique_col0 = np.unique(data[:, 0])
    unique_col1 = np.unique(data[:, 1])

    print(f"Loaded {input_file}")
    print(f"Table shape: {data.shape}")
    print(
        f"Unique values in column 0: {len(unique_col0)} -> "
        f"{unique_col0[:10]}{' ...' if len(unique_col0) > 10 else ''}"
    )
    print(
        f"Unique values in column 1: {len(unique_col1)} -> "
        f"{unique_col1[:10]}{' ...' if len(unique_col1) > 10 else ''}"
    )

    nas_curves = None

    for value_column in args.value_columns:
        energy_vals, phase_vals, phase_energy_map = infer_phase_energy_map(
            data, value_column=value_column
        )

        phase_bin_count = len(phase_vals)
        if phase_bin_count != 64:
            print(
                f"Warning: file contains {phase_bin_count} unique phase bins, not 64. "
                "The script is using the phase bins encoded in the file."
            )

        if value_column == 2:
            label = "observed_counts"
            title_prefix = "Observed Counts"
        elif value_column == 3:
            label = "model_counts"
            title_prefix = "Best-Fit Model Counts"
        elif value_column == 4:
            label = "model_counts_wo_bg"
            title_prefix = "Best-Fit Model Counts wo Background"
        else:
            label = f"column_{value_column}"
            title_prefix = f"Column {value_column}"

        print(f"Energy bins: {len(energy_vals)}")
        print(f"Phase bins: {len(phase_vals)}")
        print(f"Plotting data column: {value_column} ({title_prefix})")

        overlay_curves = None
        overlay_suffix = None
        if value_column == 3 or value_column == 4:
            if nas_curves is None:
                nas_curves = load_nas_spot_curves(args.nas_dir, phase_count=len(phase_vals))

            spot2_scale = 1.0  # <-- your factor here
            nas_total = nas_curves["spot1"] + (nas_curves["spot2"] * spot2_scale) + nas_curves["spot3"]

            overlay_curves = [
                ("NAS total", nas_total, {"color": "tab:orange", "lw": 2.0}),
                ("spot 1", nas_curves["spot1"], {"color": "tab:green", "lw": 1.0, "ls": "--"}),
                ("spot 2", nas_curves["spot2"], {"color": "tab:red", "lw": 1.0, "ls": "--"}),
                ("spot 3", nas_curves["spot3"], {"color": "tab:purple", "lw": 1.0, "ls": "--"}),
            ]
            overlay_suffix = "with_nas_spots"

        plot_phase_energy_and_bolometric(
            energy_vals=energy_vals,
            phase_vals=phase_vals,
            phase_energy_map=phase_energy_map,
            title_prefix=title_prefix,
            output_dir=output_dir,
            stem=f"j0030_{label}",
            show=not args.no_show,
            overlay_curves=overlay_curves,
            overlay_stem_suffix=overlay_suffix,
        )


if __name__ == "__main__":
    main()
