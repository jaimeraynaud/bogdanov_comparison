#!/usr/bin/env python3
"""
Plot spot3 bolometric light curves from three NAS directories (700, 1500, 436)
and compare them on the same axes. Legends are the X values.

Saves output to j0030/plots/spot3_resolution_comparison.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Paths (as requested)
MODEL_PATH = Path(__file__).parent / "data" / "j0030_phase_channel_model.txt"
# NAS_436 = Path("/Users/jraynau1/Workspace/data_analysis_visualization/j0030/data/436")
# NAS_700 = Path("/Users/jraynau1/Workspace/CLionProjects/ray_tracing/output_data/j0030/lmop150")
# NAS_1500 = Path("/Users/jraynau1/Workspace/data_analysis_visualization/j0030/data/1500")

NAS_436 = Path("/Users/jraynau1/Workspace/CLionProjects/ray_tracing/output_data/j0030/lmop150/1500")
NAS_700 = Path("/Users/jraynau1/Workspace/CLionProjects/ray_tracing/output_data/j0030/lmop150")
NAS_1500 = Path("/j0030/data/old/1500")


# NAS_436 = Path("/Users/jraynau1/Workspace/data_analysis_visualization/j0030/data/436/lmop125")
# NAS_700 = Path("/Users/jraynau1/Workspace/data_analysis_visualization/j0030/data/436/lmop150")
# NAS_1500 = Path("/Users/jraynau1/Workspace/data_analysis_visualization/j0030/data/436/lmop150/1500")

OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "spot3_resolution_comparison.png"

SHIFT_BINS = 5 #21.2  # same fractional circular shift used in j0030_photon_visualization


def fractional_circular_shift(curve: np.ndarray, shift_bins: float) -> np.ndarray:
    n = len(curve)
    freqs = np.fft.rfft(curve)
    k = np.arange(len(freqs))
    phase = np.exp(2j * np.pi * k * shift_bins / n)
    return np.fft.irfft(freqs * phase, n=n)


def infer_phase_count_from_model(model_path: Path) -> int:
    data = np.loadtxt(model_path)
    phase_vals = np.unique(data[:, 1])
    return len(phase_vals)


def load_spot3_curve(nas_dir: Path, phase_count: int) -> np.ndarray:
    # Load unfolded photon counts (before folding)
    nofolding_path = nas_dir / "spot3_photcounts_nofolding.dat"
    if nofolding_path.exists():
        mat_nofolding = np.loadtxt(nofolding_path)
        total_nofolding = np.sum(mat_nofolding)
        print(f"  Total photon counts in {nas_dir.name}/spot3_photcounts_nofolding.dat (before folding): {total_nofolding:.2f}")
    else:
        print(f"  {nofolding_path.name} not found")
    
    # Load folded photon counts
    path = nas_dir / "spot3_test_data_counts.dat"
    if not path.exists():
        raise FileNotFoundError(f"NAS spot3 file not found: {path}")
    mat = np.loadtxt(path)
    # Print total photon counts (sum over all rows and columns)
    total_counts = np.sum(mat)
    print(f"  Total photon counts in {path.parent.name}/spot3_test_data_counts.dat (after folding): {total_counts:.2f}")
    # determine axis containing phase bins
    if phase_count in mat.shape:
        if mat.shape[0] == phase_count:
            curve = mat.sum(axis=1)
        elif mat.shape[1] == phase_count:
            curve = mat.sum(axis=0)
        else:
            raise ValueError(f"Could not infer phase axis for {path} with shape {mat.shape}")
    else:
        # best guess: if one dimension equals 32 or 64 etc, try to choose
        if mat.shape[0] < mat.shape[1]:
            curve = mat.sum(axis=1)
        else:
            curve = mat.sum(axis=0)
    return curve


def main():
    phase_count = infer_phase_count_from_model(MODEL_PATH)
    print(f"Detected phase bin count from model: {phase_count}")

    dirs = [("700", NAS_700), ("1500", NAS_1500), ("436", NAS_436)]
    curves = {}

    for label, d in dirs:
        try:
            curve = load_spot3_curve(d, phase_count)
        except Exception as e:
            print(f"Error loading from {d}: {e}")
            continue
        # apply same fractional shift used elsewhere
        curve_shifted = fractional_circular_shift(curve, SHIFT_BINS)
        curves[label] = curve_shifted
        print(f"Loaded and shifted spot3 from {d}: length {len(curve_shifted)}")

    if not curves:
        raise RuntimeError("No curves loaded; aborting")

    # Build phase axis (use indices as phase axis if we don't have absolute values)
    n = phase_count
    phase_axis = (np.arange(n) + 0.5) / n

    plt.figure(figsize=(9, 4.5), dpi=150)
    for label, c in curves.items():
        if len(c) != n:
            print(f"Warning: curve {label} length {len(c)} != phase_count {n}; truncating/padding")
            if len(c) > n:
                c = c[:n]
            else:
                c = np.pad(c, (0, n - len(c)))
        plt.plot(phase_axis, c, label=f"X={label}")
        # Print total photon counts
        print(f"Total counts for {label}: {np.sum(c):.2f}")

    plt.xlabel("phase")
    plt.ylabel("summed counts (spot3)")
    plt.title("Spot3 Bolometric Light Curves: resolution comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200)
    print(f"Saved comparison figure to {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()

