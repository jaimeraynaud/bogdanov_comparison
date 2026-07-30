#!/usr/bin/env python3
"""Visualize the J0030 phase-channel model and NAS spot-count comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_FILE = Path(__file__).parent / "data" / "j0030_phase_channel_model.txt"
DEFAULT_NAS_DIR = Path(__file__).parent / "data" / "reproducing"
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
        cycles: float = 1.0,
):
    """Plot the 2D phase-energy map."""
    # Treat the phase labels as bin centers if they are integer indices.
    if np.allclose(phase_vals, np.round(phase_vals)):
        phase_axis = (phase_vals + 0.5) / len(phase_vals) * cycles
        phase_label = "phase"
    else:
        phase_axis = phase_vals * cycles
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
        cycles: float = 1.0,
        y_max: float | None = None,
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
    # Build a base phase axis that spans one cycle [0,1) (treat integer indices as bin centers)
    if np.allclose(phase_vals, np.round(phase_vals)):
        base_phase = (phase_vals + 0.5) / len(phase_vals)
        phase_label = "phase"
    else:
        base_phase = phase_vals
        phase_label = "phase"

    # If cycles is an integer >= 2, repeat the one-cycle light curve to show multiple periods
    if float(cycles).is_integer() and int(cycles) >= 2:
        n_cycles = int(round(cycles))
        phase_axis = np.concatenate([base_phase + k for k in range(n_cycles)])
        bolometric_plot = np.tile(bolometric, n_cycles)
        overlay_plot_curves = None
        if overlay_curves:
            overlay_plot_curves = []
            for label, curve, style_kwargs in overlay_curves:
                overlay_plot_curves.append((label, np.tile(curve, n_cycles), style_kwargs))
    else:
        # Non-integer cycles: scale the phase axis (backwards compatible behavior)
        phase_axis = base_phase * cycles
        bolometric_plot = bolometric
        overlay_plot_curves = overlay_curves

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)
    # ax.plot(phase_axis, bolometric_plot, color="tab:blue", lw=2.2, label="Miller's model")

    if overlay_plot_curves:
        for label, curve, style_kwargs in overlay_plot_curves:
            ax.plot(phase_axis, curve, label=label, **style_kwargs)

    ax.set_title(title)
    ax.set_xlabel(phase_label)
    ax.set_ylabel("summed counts")
    ax.set_xlim(phase_axis.min(), phase_axis.max())
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max)
    ax.legend(frameon=False)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved bolometric light curve to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

from functools import reduce

def _load_nas_spot_phase_energy_map(nas_dir: Path, spot_names, phase_shift_bins=21.25):
    """Return a (n_energy, n_phase) NAS map summed over the requested spots."""
    def fractional_circular_shift(curve, shift_bins, axis=0):
        n = curve.shape[axis]
        freqs = np.fft.rfft(curve, axis=axis)
        k = np.arange(freqs.shape[axis])
        shape = [1] * curve.ndim
        shape[axis] = -1
        phase = np.exp(2j * np.pi * k.reshape(shape) * shift_bins / n)
        return np.fft.irfft(freqs * phase, n=n, axis=axis)

    total = None
    for name in spot_names:
        path = nas_dir / f"{name}_photcounts_nofolding.dat"
        m = load_nas_spot_counts(path)          # shape: (phase, energy) per your existing code
        total = m if total is None else total + m

    total = fractional_circular_shift(total, phase_shift_bins, axis=0)
    return total.T   # -> (energy, phase), matching infer_phase_energy_map convention

def plot_energy_spectrum(
        energy_vals,
        phase_energy_map,
        *,
        title: str = "J0030 Energy Spectrum (summed over phase)",
        output_path: Path | None = None,
        show: bool = True,
        overlay_spectra=None,
        log_y: bool = True,
        primary_label: str = "best-fit model",
        primary_style: dict | None = None,
):
    """Plot a 1D spectrum: photon counts vs. energy channel.

    The 2D phase-energy map is summed along the phase axis to produce
    counts-per-channel. Additional spectra can be overlaid via
    ``overlay_spectra`` as a list of ``(label, 1D-or-2D-array, style_kwargs)``.
    A 2D overlay is automatically summed along the phase axis to match
    ``energy_vals``.
    """
    if phase_energy_map.ndim != 2:
        raise ValueError(
            f"Expected 2D phase_energy_map, got shape {phase_energy_map.shape}"
        )
    if phase_energy_map.shape[0] != len(energy_vals):
        if phase_energy_map.shape[1] == len(energy_vals):
            phase_energy_map = phase_energy_map.T
        else:
            raise ValueError(
                f"phase_energy_map shape {phase_energy_map.shape} is not "
                f"compatible with energy_vals length {len(energy_vals)}"
            )

    spectrum = phase_energy_map.sum(axis=1)  # sum over phase bins
    print(
        f"DEBUG: energy spectrum -> {spectrum.shape[0]} channels, "
        f"min={spectrum.min():.3g}, max={spectrum.max():.3g}, "
        f"total counts={spectrum.sum():.3g}"
    )

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)

    style = {"color": "tab:blue", "lw": 2.0, "drawstyle": "steps-mid"}
    if primary_style:
        style.update(primary_style)
    ax.plot(energy_vals, spectrum, label=primary_label, **style)

    # Track all curves so the log-y lower bound covers overlays too.
    all_curves = [spectrum]

    if overlay_spectra:
        for label, curve, style_kwargs in overlay_spectra:
            curve = np.asarray(curve, dtype=float)
            if curve.ndim == 2:
                if curve.shape[0] == len(energy_vals):
                    curve = curve.sum(axis=1)
                elif curve.shape[1] == len(energy_vals):
                    curve = curve.sum(axis=0)
                else:
                    raise ValueError(
                        f"Overlay '{label}' has shape {curve.shape}, "
                        f"incompatible with energy length {len(energy_vals)}"
                    )
            print(
                f"DEBUG: overlay '{label}' -> "
                f"min={curve.min():.3g}, max={curve.max():.3g}, "
                f"total={curve.sum():.3g}"
            )
            kwargs = {"drawstyle": "steps-mid"}
            kwargs.update(style_kwargs or {})
            ax.plot(energy_vals, curve, label=label, **kwargs)
            all_curves.append(curve)

    ax.set_title(title)
    ax.set_xlabel("energy channel")
    ax.set_ylabel("summed counts (over phase)")
    ax.set_xlim(np.min(energy_vals), np.max(energy_vals))
    if log_y:
        # Lower bound considers every curve we drew, not just the primary.
        positives = np.concatenate(
            [c[c > 0] for c in all_curves if np.any(c > 0)]
        ) if any(np.any(c > 0) for c in all_curves) else np.array([])
        if positives.size:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(positives.min() * 0.5, 1e-12))
    ax.legend(frameon=False)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved energy spectrum plot to {output_path}")

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
        cycles: float = 2.0,
        y_max: float = 20000.0,
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
        cycles=cycles,
    )
    plot_bolometric_light_curve(
        phase_vals=phase_vals,
        phase_energy_map=phase_energy_map,
        title=f"J0030 {title_prefix} Bolometric Light Curve",
        output_path=lc_path,
        show=show,
        overlay_curves=overlay_curves,
        cycles=cycles,
        y_max=y_max,
    )


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


def plot_nas_spot_phase_energy_map(
        nas_dir: Path,
        spot_names: tuple[str, ...] = ("spot1", "spot2"),
        model_file: Path | None = None,
        output_path: Path | None = None,
        show: bool = True,
        cycles: float = 1.0,
        phase_shift_bins: float = 21.25,
):
    """Plot the 2D energy-phase map summed directly from the NAS ``_test_data_counts.dat``
    files alongside Miller's NS-surface model from ``j0030_phase_channel_model.txt``.
    """
    total = None
    for spot_name in spot_names:
        path = nas_dir / f"{spot_name}_test_data_counts.dat"
        matrix = load_nas_spot_counts(path)
        total = matrix if total is None else total + matrix

    if total is None:
        raise ValueError("No NAS spot files were provided to plot.")

    # Files are stored with phase along rows and energy along columns; align to
    # Miller's model with the same fractional circular shift used for the
    # bolometric light curves, then transpose to (energy, phase).
    total = fractional_circular_shift(total, phase_shift_bins, axis=0)
    nas_map = total.T
    nas_energy = np.arange(nas_map.shape[0])
    nas_phase = np.arange(nas_map.shape[1])

    if model_file is None:
        model_file = DEFAULT_DATA_FILE
    model_data = load_phase_channel_table(model_file)
    # Column 4 = best-fit model components from only the NS surface.
    model_energy, model_phase, model_map = infer_phase_energy_map(model_data, value_column=4)

    def _phase_axis(phase_vals, n):
        if np.allclose(phase_vals, np.round(phase_vals)):
            return (phase_vals + 0.5) / n * cycles
        return phase_vals * cycles

    nas_phase_axis = _phase_axis(nas_phase, len(nas_phase))
    model_phase_axis = _phase_axis(model_phase, len(model_phase))

    fig, (ax_nas, ax_model, ax_diff) = plt.subplots(
        1, 3, figsize=(21, 6), dpi=180, constrained_layout=True
    )

    # Shared color scale so the two maps are visually comparable.
    vmin = float(min(nas_map.min(), model_map.min()))
    vmax = float(max(nas_map.max(), model_map.max()))

    nas_mesh = ax_nas.pcolormesh(
        nas_phase_axis, nas_energy, nas_map,
        shading="auto", cmap="magma", vmin=vmin, vmax=vmax,
    )
    ax_nas.set_title(f"J0030 NAS {'+'.join(spot_names)} Phase-Energy Map")
    ax_nas.set_xlabel("phase")
    ax_nas.set_ylabel("energy channel")
    ax_nas.set_xlim(nas_phase_axis.min(), nas_phase_axis.max())
    ax_nas.set_ylim(nas_energy.min(), nas_energy.max())

    model_mesh = ax_model.pcolormesh(
        model_phase_axis, model_energy, model_map,
        shading="auto", cmap="magma", vmin=vmin, vmax=vmax,
    )
    fig.colorbar(model_mesh, ax=(ax_nas, ax_model), label="counts")
    ax_model.set_title("Miller NS-surface Model Phase-Energy Map")
    ax_model.set_xlabel("phase")
    ax_model.set_ylabel("energy channel")
    ax_model.set_xlim(model_phase_axis.min(), model_phase_axis.max())
    ax_model.set_ylim(model_energy.min(), model_energy.max())

    # Ratio Miller / NAS, on the overlapping shape.
    n_energy = min(nas_map.shape[0], model_map.shape[0])
    n_phase = min(nas_map.shape[1], model_map.shape[1])
    nas_clip = nas_map[:n_energy, :n_phase]
    model_clip = model_map[:n_energy, :n_phase]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nas_clip != 0, model_clip / nas_clip, np.nan)

    diff_phase_axis = _phase_axis(np.arange(n_phase), n_phase)
    diff_energy = np.arange(n_energy)

    diff_mesh = ax_diff.pcolormesh(
        diff_phase_axis, diff_energy, ratio,
        shading="auto", cmap="magma",
        vmin=1.0, vmax=1.3,
    )
    fig.colorbar(diff_mesh, ax=ax_diff, label="Miller / NAS")
    ax_diff.set_title("Miller / NAS ratio")
    ax_diff.set_xlabel("phase")
    ax_diff.set_ylabel("energy channel")
    ax_diff.set_xlim(diff_phase_axis.min(), diff_phase_axis.max())
    ax_diff.set_ylim(diff_energy.min(), diff_energy.max())

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved phase-energy map comparison to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def load_nas_spot_curves(nas_dir: Path, phase_count: int) -> dict[str, np.ndarray]:
    """Load NAS spot count matrices and convert them to phase-bolometric curves."""
    spot_curves: dict[str, np.ndarray] = {}
    missing: list[str] = []

    # Helper: fractional circular shift used to align NAS curves to the model
    def fractional_circular_shift(curve, shift_bins):
        n = len(curve)
        freqs = np.fft.rfft(curve)
        k = np.arange(len(freqs))
        phase = np.exp(2j * np.pi * k * shift_bins / n)
        return np.fft.irfft(freqs * phase, n=n)

    for spot_name in ("spot1", "spot2", "spot3"):
        path = nas_dir / f"{spot_name}_test_data_counts.dat"

        if not path.exists():
            # If the file is missing, create a zero curve so downstream code
            # that expects 'spot1','spot2','spot3' keys won't crash.
            print(f"Warning: NAS spot file not found: {path}. Treating {spot_name} as zero.")
            spot_curves[spot_name] = np.zeros(phase_count)
            missing.append(spot_name)
            continue

        # Attempt to load the matrix; if loading fails, warn and substitute zeros.
        try:
            matrix = load_nas_spot_counts(path)
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}. Treating {spot_name} as zero.")
            spot_curves[spot_name] = np.zeros(phase_count)
            missing.append(spot_name)
            continue

        # Validate that the matrix contains the expected phase axis
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

        # Apply the same phase shift as the model (fractional shift allowed)
        curve = fractional_circular_shift(curve, 21.25)

        spot_curves[spot_name] = curve

    if missing:
        print(f"Note: the following NAS spot files were missing or failed to load: {', '.join(missing)}")

    return spot_curves


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the J0030 phase-channel model as a 2D map and bolometric curve."
    )
    parser.add_argument(
        "--input",type=Path,
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
    parser.add_argument(
        "--cycles",
        type=float,
        default=2.0,
        help="Number of cycles to display on the phase axis (default: 2.0).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=20000,
        help="Maximum y value for bolometric plots (default: 20000).",
    )
    parser.add_argument(
        "--observed-offset",
        type=float,
        default=0.0,
        help="Constant offset to add to the observed bolometric curve (default: 0.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Print effective runtime plotting settings so IDE runs without CLI args are
    # easy to confirm (PyCharm often runs files without passing arguments).
    print(f"Runtime plotting settings: cycles={args.cycles}, y_max={args.y_max}")
    if args.cycles <= 0:
        raise ValueError("--cycles must be > 0")
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

    # Plot the 2D energy-phase map directly from the NAS spot1+spot2 _test_data_counts.dat files.
    plot_nas_spot_phase_energy_map(
        nas_dir=args.nas_dir,
        spot_names=("spot1", "spot2"),
        output_path=output_dir / "j0030_nas_spots_phase_energy_map.png",
        show=not args.no_show,
        cycles=args.cycles,
    )

    # Load observed counts (column 2) and background (column 5) early so it can be added as an overlay
    observed_energy_vals, observed_phase_vals, observed_phase_energy_map = infer_phase_energy_map(
        data, value_column=2
    )
    background_energy_vals, background_phase_vals, background_phase_energy_map = infer_phase_energy_map(
        data, value_column=5
    )
    observed_bolometric = observed_phase_energy_map.sum(axis=0)
    background_bolometric = background_phase_energy_map.sum(axis=0)
    # Subtract background from observed counts
    observed_bolometric = observed_bolometric  # - background_bolometric
    print(
        f"DEBUG: observed_bolometric (after bg subtraction) shape={observed_bolometric.shape}, "
        f"min={observed_bolometric.min():.2f}, max={observed_bolometric.max():.2f}"
    )

    # Load Miller's NS-surface model (column 4) once for use as an overlay.
    miller_energy_vals, miller_phase_vals, miller_phase_energy_map = infer_phase_energy_map(
        data, value_column=4
    )
    miller_bolometric = miller_phase_energy_map.sum(axis=0)
    print(
        f"DEBUG: miller_bolometric shape={miller_bolometric.shape}, "
        f"min={miller_bolometric.min():.2f}, max={miller_bolometric.max():.2f}, "
        f"total={miller_bolometric.sum():.2f}"
    )

    for value_column in args.value_columns:
        print(f"Processing column {value_column} with cycles={args.cycles} y_max={args.y_max}")
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

            add_factor = 0
            multiply_factor = 1.0  # <-- your factor here
            nas_total = (nas_curves["spot1"] + nas_curves["spot2"]) * multiply_factor + add_factor

            overlay_curves = [
                ("Our model",      nas_total,         {"color": "tab:orange", "lw": 2.0}),
                ("Miller's model", miller_bolometric, {"color": "tab:blue",   "lw": 2.0, "ls": "-"}),
            ]
            print(f"DEBUG: overlay_curves has {len(overlay_curves)} curves")
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
            cycles=args.cycles,
            y_max=args.y_max,
        )

        # ---- 1D energy spectrum (sum over phase) ----
        # Pull the model components and observed/background maps for overlays.
        model_energy_vals, _, model_phase_energy_map = infer_phase_energy_map(
            data, value_column=3
        )
        nssurf_energy_vals, _, nssurf_phase_energy_map = infer_phase_energy_map(
            data, value_column=4
        )

        # Observed counts minus phase-independent background (from j0030_phase_channel_model.txt).
        # Both maps share the same (energy, phase) shape, so a direct subtraction is well-defined.
        obs_minus_bg_map = observed_phase_energy_map - background_phase_energy_map
        print(
            f"DEBUG: obs - bg map shape={obs_minus_bg_map.shape}, "
            f"min={obs_minus_bg_map.min():.2f}, max={obs_minus_bg_map.max():.2f}, "
            f"total={obs_minus_bg_map.sum():.2f}"
        )

        # NAS spot1+spot2 model in (energy, phase) form.
        nas_map = _load_nas_spot_phase_energy_map(args.nas_dir, ("spot1", "spot2"))
        nas_energy_vals = np.arange(nas_map.shape[0])

        # Sanity check: the NAS energy axis length should match the model file's energy axis,
        # otherwise the overlays can't share the same x-axis.
        if len(nas_energy_vals) != len(observed_energy_vals):
            print(
                f"WARNING: NAS energy length ({len(nas_energy_vals)}) differs from "
                f"model file energy length ({len(observed_energy_vals)}). "
                "Overlays will be plotted against the NAS axis only where lengths match."
            )

        plot_energy_spectrum(
            energy_vals=nas_energy_vals,
            phase_energy_map=nas_map,
            primary_label="Our spot1+spot2 model",
            primary_style={"color": "tab:blue", "lw": 2.0},
            title="J0030 Energy Spectrum: Our spot1+spot2 vs. observed - background",
            output_path=output_dir / "j0030_nas_spots_energy_spectrum.png",
            show=not args.no_show,
            overlay_spectra=[
                # ("observed - background", obs_minus_bg_map,        {"color": "black",      "lw": 1.5}),
                # Optional extra context curves -- comment out if you only want the two:
                # ("best-fit model",        model_phase_energy_map,  {"color": "tab:orange", "lw": 1.5, "ls": "--"}),
                # ("NS-surface only",       nssurf_phase_energy_map, {"color": "tab:cyan",   "lw": 1.2, "ls": ":"}),
            ],
            log_y=True,
        )


if __name__ == "__main__":
    main()