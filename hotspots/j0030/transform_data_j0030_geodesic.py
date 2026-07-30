#!/usr/bin/env python3
"""
Transform hotspot data and generate oval hotspot grids (3 spots).

Updated to:
- Support 3 hotspots
- Use oval geometry based on (Δθ, f scaling)
"""

import numpy as np
from pathlib import Path


CASE_NAME = "j0030"


def angular_distance_on_sphere(theta1, phi1, theta2, phi2):
    cos_distance = (np.cos(theta1) * np.cos(theta2) +
                    np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2))
    cos_distance = np.clip(cos_distance, -1.0, 1.0)
    return np.arccos(cos_distance)


def delta_phi(phi, phi_c):
    """Shortest wrapped angular difference."""
    dphi = phi - phi_c
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def transform_hotspot_data(input_file, output_file, expected_shape=None):
    print(f"Reading {input_file}...")
    data = np.loadtxt(input_file)

    if expected_shape is not None and data.shape != expected_shape:
        print(f"Warning: Expected shape {expected_shape}, got {data.shape}")

    n_phi = data.shape[0] - 1
    n_theta = data.shape[1] - 1

    dph = 2.0 * np.pi / n_phi
    dcth = 2.0 / n_theta

    phi_values = []
    theta_values = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 1:
                phi = i * dph
                cos_theta = -1.0 + j * dcth
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                phi_values.append(phi)
                theta_values.append(theta)

    output_data = np.column_stack((phi_values, theta_values))
    np.savetxt(output_file, output_data, fmt='%.18e', delimiter=' ')

    print(f"Saved {len(phi_values)} points → {output_file}")
    return None


def generate_paper_hotspots_grid(grid_resolution=600, output_dir=None):

    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    # ===============================
    # NEW: 3 OVAL HOTSPOTS
    # ===============================
    spots = [
        {
            'name': 'spot1',
            'theta_c': 2.330,
            'dtheta': 0.032,
            'phi_c': 2.61799,
            'f': 5.335,
        },
        {
            'name': 'spot2',
            'theta_c': 2.446,
            'dtheta': 0.029,
            'phi_c': 2.61799 - 2 * np.pi * 0.463,
            'f': 16.588,
        },
        {
            'name': 'spot3',
            'theta_c': 3.056,
            'dtheta': 0.087,
            'phi_c': 2.61799 - 2 * np.pi * 0.427,
            'f': 1.253,
        }
    ]
    # spots = [
    #     {
    #         'name': 'spot3',
    #         'theta_c': 3.056,
    #         'dtheta': 0.087,
    #         'phi_c': 2 * np.pi * 0.427,
    #         'f': 1.253,
    #     }
    # ]

    print("=" * 60)
    print("GENERATING OVAL HOTSPOT GRIDS (3 SPOTS)")
    print("=" * 60)

    # n_phi = grid_resolution - 1
    # n_theta = grid_resolution - 1
    # dph = 2.0 * np.pi / n_phi
    # dcth = 2.0 / n_theta

    # combined_grid = np.zeros((grid_resolution, grid_resolution)) # goes from 0..grid_resolution - 1

    # to review
    n_phi = grid_resolution
    n_theta = grid_resolution
    dph = 2.0 * np.pi / n_phi
    dcth = 2.0 / n_theta

    combined_grid = np.zeros((grid_resolution + 1, grid_resolution + 1)) # goes from 0..grid_resolution

    for spot in spots:

        print(f"\nProcessing {spot['name']}")
        print(f"θc={spot['theta_c']:.3f}, Δθ={spot['dtheta']:.3f}, f={spot['f']}")

        grid = np.zeros((grid_resolution + 1, grid_resolution + 1))
        point_count = 0

        theta_c = spot['theta_c']
        phi_c = spot['phi_c']
        dtheta = spot['dtheta']
        f = spot['f']

        sin_tc = np.sin(theta_c)

        if sin_tc < 1e-8:
            print("Skipping (too close to pole)")
            continue

        # Circular equivalent longitudinal extent
        dphi_circ = dtheta / sin_tc

        # Apply scaling
        dphi = f * dphi_circ

        # Enforce max width ≤ π
        dphi = min(dphi, np.pi)

        # for i in range(grid_resolution):
        #     for j in range(grid_resolution):
        for i in range(grid_resolution + 1): # to review
            for j in range(grid_resolution + 1):

                phi = i * dph
                cos_theta = -1.0 + j * dcth
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                dt = (theta - theta_c) / dtheta
                dp = delta_phi(phi, phi_c) / dphi

                if dt**2 + dp**2 <= 1.0:
                    grid[i, j] = 1.0
                    point_count += 1

        combined_grid = np.maximum(combined_grid, grid)

        output_file = output_dir / f"hotspot_{spot['name']}_{grid_resolution}_newconvention.dat"
        np.savetxt(output_file, grid, fmt='%.18e', delimiter=' ')

        coverage = point_count / (grid_resolution**2) * 100
        print(f"Points: {point_count} ({coverage:.2f}%)")
        print(f"Saved → {output_file.name}")

    combined_file = output_dir / f"hotspot_combined_{grid_resolution}.dat"
    np.savetxt(combined_file, combined_grid, fmt='%.18e', delimiter=' ')

    print("\nCombined grid saved → hotspot_combined.dat")
    print("=" * 60)


def main():

    base_path = Path(__file__).parent
    case_dir = base_path / "hotspots"

    generate_paper_hotspots_grid(
        grid_resolution=5000,
        output_dir=case_dir
    )


if __name__ == "__main__":
    main()