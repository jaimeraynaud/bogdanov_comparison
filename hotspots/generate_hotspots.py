#!/usr/bin/env python3
"""
Generate oval hotspot grids for supported cases.

Cases:
- j0030: oval hotspot geometry (dtheta, f)
- bogdanov, j0740: circular hotspots using angular distance
"""

import argparse
import numpy as np
from pathlib import Path


def delta_phi(phi, phi_c):
    """Shortest wrapped angular difference."""
    dphi = phi - phi_c
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def angular_distance_on_sphere(theta1, phi1, theta2, phi2):
    """Angular distance on the sphere using spherical law of cosines."""
    cos_distance = (
        np.cos(theta1) * np.cos(theta2)
        + np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
    )
    return np.arccos(np.clip(cos_distance, -1.0, 1.0))


def generate_paper_hotspots_grid(spots, grid_resolution=600, output_dir=None, file_tag=""):
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    print("=" * 60)
    print("GENERATING HOTSPOT GRIDS")
    print("=" * 60)

    n_phi = grid_resolution
    n_theta = grid_resolution
    dph = 2.0 * np.pi / n_phi
    dcth = 2.0 / n_theta

    combined_grid = np.zeros((grid_resolution + 1, grid_resolution + 1))

    for spot in spots:
        print(f"\nProcessing {spot['name']}")
        if spot["geometry"] == "oval":
            print(
                f"Geometry: oval | θc={spot['theta_c']:.3f}, "
                f"Δθ={spot['dtheta']:.3f}, f={spot['f']}"
            )
        else:
            print(
                f"Geometry: circular | θc={spot['theta_c']:.3f}, "
                f"radius={spot['radius']:.3f}"
            )

        grid = np.zeros((grid_resolution + 1, grid_resolution + 1))
        point_count = 0

        theta_c = spot["theta_c"]
        phi_c = spot["phi_c"]

        if spot["geometry"] == "oval":
            dtheta = spot["dtheta"]
            f = spot["f"]

            sin_tc = np.sin(theta_c)
            if sin_tc < 1e-8:
                print("Skipping (too close to pole)")
                continue

            dphi_circ = dtheta / sin_tc
            dphi = f * dphi_circ
            dphi = min(dphi, np.pi)
        else:
            radius = spot["radius"]

        for i in range(grid_resolution + 1):
            for j in range(grid_resolution + 1):
                phi = i * dph
                cos_theta = -1.0 + j * dcth
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                if spot["geometry"] == "oval":
                    dt = (theta - theta_c) / dtheta
                    dp = delta_phi(phi, phi_c) / dphi
                    inside = dt**2 + dp**2 <= 1.0
                else:
                    distance = angular_distance_on_sphere(theta, phi, theta_c, phi_c)
                    inside = distance <= radius

                if inside:
                    grid[i, j] = 1.0
                    point_count += 1

        combined_grid = np.maximum(combined_grid, grid)

        output_file = output_dir / f"hotspot_{spot['name']}_{grid_resolution}{file_tag}.dat"
        np.savetxt(output_file, grid, fmt="%.18e", delimiter=" ")

        coverage = point_count / ((grid_resolution + 1) ** 2) * 100
        print(f"Points: {point_count} ({coverage:.2f}%)")
        print(f"Saved → {output_file.name}")

    combined_file = output_dir / f"hotspot_combined_{grid_resolution}{file_tag}.dat"
    np.savetxt(combined_file, combined_grid, fmt="%.18e", delimiter=" ")

    print(f"\nCombined grid saved → {combined_file.name}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper hotspot grids by case.")
    parser.add_argument(
        "--case",
        choices=["bogdanov", "j0740", "j0030"],
        default="j0030",
        help="Case name to generate hotspots for (default: j0030).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated grids (default: case folder).",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=None,
        help="Override default grid resolution for the selected case.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    case_configs = {
        "j0030": {
            "grid_resolution": 7000,
            "file_tag": "_twohotspots_j0030",
            "spots": [
                {
                    "name": "spot1",
                    "geometry": "oval",
                    "theta_c": 2.232,
                    "dtheta": 0.031,
                    "phi_c": 2.61799,
                    "f": 6.024,
                },
                {
                    "name": "spot2",
                    "geometry": "oval",
                    "theta_c": 2.394,
                    "dtheta": 0.029,
                    "phi_c": 2.61799 - 2 * np.pi * 0.458,
                    "f": 17.744,
                },
            ],
        },
        "bogdanov": {
            "grid_resolution": 2000,
            "file_tag": "_bogdanov",
            "spots": [
                {
                    "name": "spot1",
                    "geometry": "circular",
                    "theta_c": 0.6283,
                    "phi_c": 0.827731,
                    "radius": 0.01,
                },
                {
                    "name": "spot2",
                    "geometry": "circular",
                    "theta_c": 2.077,
                    "phi_c": 3.577958,
                    "radius": 0.33,
                },
            ],
        },
        "j0740": {
            "grid_resolution": 5000,
            "file_tag": "_j0740",
            "spots": [
                {
                    "name": "spot1",
                    "geometry": "circular",
                    "theta_c": 1.387,
                    "phi_c": 0.0,
                    "radius": 0.092,
                },
                {
                    "name": "spot2",
                    "geometry": "circular",
                    "theta_c": 1.98,
                    "phi_c": 3.5696,
                    "radius": 0.112,
                },
            ],
        },
    }

    config = case_configs[args.case]
    grid_resolution = args.grid_resolution or config["grid_resolution"]

    base_path = Path(__file__).parent
    default_output_dir = base_path / args.case
    output_dir = args.output_dir or default_output_dir

    generate_paper_hotspots_grid(
        spots=config["spots"],
        grid_resolution=grid_resolution,
        output_dir=output_dir,
        file_tag=config["file_tag"],
    )


if __name__ == "__main__":
    main()

