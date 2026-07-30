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
    """True geodesic distance on a sphere (in radians)."""
    cos_distance = (np.cos(theta1) * np.cos(theta2) +
                    np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2))
    cos_distance = np.clip(cos_distance, -1.0, 1.0)
    return np.arccos(cos_distance)


def delta_phi(phi, phi_c):
    """Shortest wrapped angular difference."""
    dphi = phi - phi_c
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def tangent_plane_distance(theta, phi, theta_c, phi_c, dtheta, f):
    """
    Compute elliptical distance in tangent plane coordinates at hotspot center.
    
    This approach is pole-safe because it works in local (ξ, η) coordinates
    where the pole singularity is naturally handled.
    
    ξ = θ - θ_c  (meridional displacement)
    η = sin(θ_c) * Δφ  (zonal displacement, pre-scaled by sin(θ_c))
    
    Returns elliptical distance metric: (ξ/a)² + (η/b)²
    where a ~ Δθ and b ~ Δθ/f (zonal semi-axis is smaller if f > 1)
    """
    sin_tc = np.sin(theta_c)
    
    # Meridional distance (always safe)
    xi = theta - theta_c
    
    # Zonal distance in tangent plane (phi is pre-scaled by sin(theta_c))
    # This eliminates the 1/sin(theta_c) blow-up at the pole
    dphi_wrapped = delta_phi(phi, phi_c)
    eta = sin_tc * dphi_wrapped
    
    # Ellipse semi-axes in tangent plane
    # a = Δθ (meridional extent)
    # b = Δθ * f (zonal extent after scaling by f)
    a = dtheta
    b = dtheta * f
    
    # Elliptical distance (normalized)
    dist_sq = (xi / a)**2 + (eta / b)**2
    
    return dist_sq


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


def generate_paper_hotspots_grid_geodesic(grid_resolution=600, output_dir=None, use_method="tangent_plane"):
    """
    Generate oval hotspot grids using geodesic-safe methods.
    
    use_method: "tangent_plane" (recommended) or "cartesian" (old method)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

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

    print("=" * 80)
    print(f"GENERATING OVAL HOTSPOT GRIDS - METHOD: {use_method.upper()}")
    print("=" * 80)

    n_phi = grid_resolution
    n_theta = grid_resolution
    dph = 2.0 * np.pi / n_phi
    dcth = 2.0 / n_theta

    combined_grid = np.zeros((grid_resolution + 1, grid_resolution + 1))

    for spot in spots:
        print(f"\n{'='*80}")
        print(f"Processing {spot['name']}")
        print(f"θc={spot['theta_c']:.4f}, Δθ={spot['dtheta']:.4f}, f={spot['f']:.4f}")
        print(f"φc={spot['phi_c']:.4f}")
        print(f"sin(θc)={np.sin(spot['theta_c']):.6f}")

        grid = np.zeros((grid_resolution + 1, grid_resolution + 1))
        point_count = 0
        theta_min, theta_max = np.pi, 0
        phi_min, phi_max = 2*np.pi, 0

        theta_c = spot['theta_c']
        phi_c = spot['phi_c']
        dtheta = spot['dtheta']
        f = spot['f']

        for i in range(grid_resolution + 1):
            for j in range(grid_resolution + 1):
                phi = i * dph
                cos_theta = -1.0 + j * dcth
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                if use_method == "tangent_plane":
                    # Pole-safe tangent plane method
                    dist_sq = tangent_plane_distance(theta, phi, theta_c, phi_c, dtheta, f)
                    is_inside = dist_sq <= 1.0
                else:
                    # Old Cartesian method (for comparison)
                    sin_tc = np.sin(theta_c)
                    if sin_tc < 1e-8:
                        is_inside = False
                    else:
                        dphi_circ = dtheta / sin_tc
                        dphi = min(f * dphi_circ, np.pi)
                        dt = (theta - theta_c) / dtheta
                        dp = delta_phi(phi, phi_c) / dphi
                        is_inside = dt**2 + dp**2 <= 1.0

                if is_inside:
                    grid[i, j] = 1.0
                    point_count += 1
                    theta_min = min(theta_min, theta)
                    theta_max = max(theta_max, theta)
                    phi_min = min(phi_min, phi)
                    phi_max = max(phi_max, phi)

        combined_grid = np.maximum(combined_grid, grid)

        output_file = output_dir / f"hotspot_{spot['name']}_{grid_resolution}_geodesic.dat"
        np.savetxt(output_file, grid, fmt='%.18e', delimiter=' ')

        coverage = point_count / (grid_resolution**2) * 100
        print(f"\nPoints: {point_count} ({coverage:.2f}%)")
        print(f"θ range: [{theta_min:.4f}, {theta_max:.4f}]")
        print(f"φ range: [{phi_min:.4f}, {phi_max:.4f}]")
        print(f"Saved → {output_file.name}")

    combined_file = output_dir / f"hotspot_combined_{grid_resolution}_geodesic.dat"
    np.savetxt(combined_file, combined_grid, fmt='%.18e', delimiter=' ')

    print(f"\n{'='*80}")
    print("Combined grid saved → hotspot_combined_geodesic.dat")
    print("=" * 80)


def generate_paper_hotspots_grid(grid_resolution=600, output_dir=None):
    """
    DEPRECATED: Use generate_paper_hotspots_grid_geodesic instead.
    This old method has pole singularity issues.
    """
    # ...existing code...
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

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

    print("=" * 60)
    print("GENERATING OVAL HOTSPOT GRIDS (3 SPOTS) - OLD CARTESIAN METHOD")
    print("=" * 60)

    n_phi = grid_resolution
    n_theta = grid_resolution
    dph = 2.0 * np.pi / n_phi
    dcth = 2.0 / n_theta

    combined_grid = np.zeros((grid_resolution + 1, grid_resolution + 1))

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

        dphi_circ = dtheta / sin_tc
        dphi = f * dphi_circ
        dphi = min(dphi, np.pi)

        for i in range(grid_resolution + 1):
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

        output_file = output_dir / f"hotspot_{spot['name']}_{grid_resolution}_cartesian.dat"
        np.savetxt(output_file, grid, fmt='%.18e', delimiter=' ')

        coverage = point_count / (grid_resolution**2) * 100
        print(f"Points: {point_count} ({coverage:.2f}%)")
        print(f"Saved → {output_file.name}")

    combined_file = output_dir / f"hotspot_combined_{grid_resolution}_cartesian.dat"
    np.savetxt(combined_file, combined_grid, fmt='%.18e', delimiter=' ')

    print("\nCombined grid saved")
    print("=" * 60)


def main():
    base_path = Path(__file__).parent
    case_dir = base_path / "hotspots"

    print("\n" + "="*80)
    print("HOTSPOT GRID GENERATION - GEODESIC VS CARTESIAN COMPARISON")
    print("="*80 + "\n")

    grid_res = 5000

    # Generate using the new GEODESIC method (pole-safe)
    print("\n>>> GEODESIC METHOD (POLE-SAFE) <<<\n")
    generate_paper_hotspots_grid_geodesic(
        grid_resolution=grid_res,
        output_dir=case_dir,
        use_method="tangent_plane"
    )

    # Generate using old CARTESIAN method (for comparison)
    print("\n\n>>> CARTESIAN METHOD (OLD - HAS POLE ISSUES) <<<\n")
    generate_paper_hotspots_grid(
        grid_resolution=grid_res,
        output_dir=case_dir
    )

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print("\nOutput files generated:")
    print("  - hotspot_spot*_*_geodesic.dat    (RECOMMENDED: pole-safe)")
    print("  - hotspot_spot*_*_cartesian.dat   (old method)")
    print("\nTo compare results:")
    print("  spot3 width (geodesic) should now be > spot2 width")
    print("  spot3 width (cartesian) was abnormally narrow (~38 cols)")


if __name__ == "__main__":
    main()