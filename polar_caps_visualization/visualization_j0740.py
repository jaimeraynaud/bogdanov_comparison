#!/usr/bin/env python3
"""Visualize J0740 hotspot grids in theta-phi coordinates."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


PAPER_THETA_CENTERS = {
    "J0740 Spot 1": 1.387,
    "J0740 Spot 2": 1.980,
}

PAPER_DELTA_THETA = {
    "J0740 Spot 1": 0.092,
    "J0740 Spot 2": 0.112,
}

PAPER_SPOT1_PHI_CYCLES = 0.0
PAPER_LONGITUDINAL_OFFSET_CYCLES = 0.428


def paper_phi_centers():
    """Return paper phi centers (radians) for both spots from cycle offsets."""
    phi1 = (PAPER_SPOT1_PHI_CYCLES % 1.0) * (2.0 * np.pi)
    phi2 = ((PAPER_SPOT1_PHI_CYCLES + PAPER_LONGITUDINAL_OFFSET_CYCLES) % 1.0) * (2.0 * np.pi)
    return {
        "J0740 Spot 1": phi1,
        "J0740 Spot 2": phi2,
    }


def grid_to_phi_theta(data, threshold=0.5):
    """Convert a hotspot grid into arrays of phi and theta hotspot coordinates."""
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {data.shape}")

    n_phi = data.shape[0] - 1
    n_theta = data.shape[1] - 1
    if n_phi <= 0 or n_theta <= 0:
        raise ValueError(f"Grid must be at least 2x2, got {data.shape}")

    hotspot_idx = np.argwhere(data > threshold)
    if hotspot_idx.size == 0:
        return np.array([]), np.array([])

    i = hotspot_idx[:, 0]
    j = hotspot_idx[:, 1]

    phi = i * (2.0 * np.pi / n_phi)
    cos_theta = -1.0 + j * (2.0 / n_theta)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return phi, theta


def load_spot_file(path, expected_shape=(5000, 5000)):
    """Load one J0740 spot grid and return phi/theta hotspot points."""
    data = np.loadtxt(path)
    if data.shape != expected_shape:
        print(f"Warning: Expected shape {expected_shape} for {path.name}, got {data.shape}")
    return grid_to_phi_theta(data)


def circular_mean_angle(angles):
    """Return circular mean for angles in radians, wrapped to [0, 2*pi)."""
    if angles.size == 0:
        return np.nan
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return mean_angle


def plot_j0740(directory, point_size=6.0, save_path=None, show=True):
    """Plot theta vs phi for the two J0740 hotspot files."""
    directory = Path(directory)
    # spot_files = [
    #     directory / "j0740_bestfit_NICERandXMM_spot1_2kres.dat",
    #     directory / "j0740_bestfit_NICERandXMM_spot2_2kres_shiftoppos.dat",
    # ]
    # spot_files = [
    #     directory / "j0740_hotspot_spot1_5k.dat",
    #     directory / "j0740_hotspot_spot2_5k.dat",
    # ]
    spot_files = [
        directory / "test_hotspot_spot1_600.dat",
        directory / "test_hotspot_spot2_600.dat",
    ]

    titles = ["J0740 Spot 1", "J0740 Spot 2"]
    colors = ["tab:orange", "tab:blue"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    phi_centers = paper_phi_centers()

    for spot_file, title, color in zip(spot_files, titles, colors):
        if not spot_file.exists():
            raise FileNotFoundError(f"Missing input file: {spot_file}")

        phi, theta = load_spot_file(spot_file)
        ax.scatter(
            phi,
            theta,
            s=point_size,
            color=color,
            alpha=0.9,
            edgecolors="none",
            label=f"{title} (n={phi.size})",
        )

        center_phi = phi_centers[title]
        center_theta = PAPER_THETA_CENTERS[title]
        delta_theta = PAPER_DELTA_THETA[title]
        ax.scatter(
            center_phi,
            center_theta,
            s=90,
            color="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
            label=f"{title} center",
        )
        ax.errorbar(
            center_phi,
            center_theta,
            yerr=delta_theta,
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
            capthick=1.1,
            zorder=4,
            label=f"{title} $\\Delta\\theta$={delta_theta:.3f}",
        )

    ax.set_title("J0740 Hotspot Grid: Theta vs Phi")
    ax.set_xlabel("phi [rad]")
    ax.set_ylabel("theta [rad]")
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, np.pi)
    # Denser coordinate lines: major every pi/4, minor every pi/8.
    ax.xaxis.set_major_locator(MultipleLocator(np.pi / 4.0))
    ax.xaxis.set_minor_locator(MultipleLocator(np.pi / 8.0))
    ax.yaxis.set_major_locator(MultipleLocator(np.pi / 4.0))
    ax.yaxis.set_minor_locator(MultipleLocator(np.pi / 8.0))
    ax.grid(which="major", alpha=0.45, linewidth=0.8)
    ax.grid(which="minor", alpha=0.25, linewidth=0.5, linestyle="--")
    ax.legend(loc="best")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    script_dir = Path(__file__).parent
    default_dir = script_dir / "j0740"

    parser = argparse.ArgumentParser(description="Plot J0740 600x600 hotspot grids as theta vs phi.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=default_dir,
        help=f"Directory containing J0740 .dat files (default: {default_dir})",
    )
    parser.add_argument("--point-size", type=float, default=6.0, help="Scatter point size.")
    parser.add_argument("--save", type=Path, default=None, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_j0740(
        directory=args.directory,
        point_size=args.point_size,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

