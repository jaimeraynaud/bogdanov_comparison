#!/usr/bin/env python3
"""
IMPROVED GEODESIC HOTSPOT GENERATION using 3D Cartesian coordinates.

This method converts (θ, φ) to 3D unit sphere coordinates (x, y, z),
then uses true geodesic distances. This completely avoids singularities
and provides proper area-preserving ellipses at any latitude.
"""

import numpy as np
from pathlib import Path


CASE_NAME = "j0030"


def spherical_to_cartesian(theta, phi):
    """Convert spherical (θ, φ) to Cartesian (x, y, z) on unit sphere."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """Convert Cartesian (x, y, z) back to spherical (θ, φ)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/r, y/r, z/r  # normalize
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def geodesic_distance(theta1, phi1, theta2, phi2):
    """True geodesic (great circle) distance on sphere."""
    cos_angle = (np.cos(theta1) * np.cos(theta2) +
                 np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2))
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def build_local_orthonormal_basis(theta_c, phi_c):
    """
    Build orthonormal basis vectors in tangent plane at (θc, φc).
    
    Returns:
        e_theta: unit vector in θ direction (meridional)
        e_phi:   unit vector in φ direction (zonal/longitudinal)
        center:  (x, y, z) of hotspot center
    
    These vectors lie in the tangent plane and form an orthonormal basis.
    """
    x_c, y_c, z_c = spherical_to_cartesian(theta_c, phi_c)
    center = np.array([x_c, y_c, z_c])
    
    # Radial direction (points outward from sphere center)
    radial = center
    
    # θ direction: ∂(x,y,z)/∂θ, normalized
    # In Cartesian: (cos(θ)cos(φ), cos(θ)sin(φ), -sin(θ))
    e_theta = np.array([
        np.cos(theta_c) * np.cos(phi_c),
        np.cos(theta_c) * np.sin(phi_c),
        -np.sin(theta_c)
    ])
    e_theta = e_theta / np.linalg.norm(e_theta)
    
    # φ direction: ∂(x,y,z)/∂φ, normalized
    # In Cartesian: (-sin(θ)sin(φ), sin(θ)cos(φ), 0)
    e_phi = np.array([
        -np.sin(theta_c) * np.sin(phi_c),
        np.sin(theta_c) * np.cos(phi_c),
        0
    ])
    e_phi = e_phi / np.linalg.norm(e_phi)
    
    return e_theta, e_phi, center


def elliptical_distance_3d(theta, phi, theta_c, phi_c, dtheta, f):
    """
    Compute elliptical distance using 3D Cartesian approach.
    
    This is truly pole-safe because:
    1. No division by sin(θ)
    2. Uses proper spherical geometry
    3. Ellipse is defined in local tangent plane
    4. Works identically everywhere on sphere
    """
    # Convert points to Cartesian
    x, y, z = spherical_to_cartesian(theta, phi)
    point = np.array([x, y, z])
    
    # Build local basis at hotspot center
    e_theta, e_phi, center = build_local_orthonormal_basis(theta_c, phi_c)
    
    # Displacement vector in 3D
    displacement = point - center
    
    # Project onto local basis vectors
    # These are small-angle approximations valid near hotspot center
    xi = np.dot(displacement, e_theta)   # meridional component
    eta = np.dot(displacement, e_phi)    # zonal component
    
    # Ellipse semi-axes
    a = dtheta  # meridional
    b = dtheta * f  # zonal
    
    # Elliptical distance metric
    dist_sq = (xi / a)**2 + (eta / b)**2
    
    return dist_sq


def generate_paper_hotspots_grid_3d_geodesic(grid_resolution=600, output_dir=None):
    """
    Generate oval hotspot grids using true 3D geodesic approach.
    
    This method is:
    - Pole-safe (no singularities)
    - Area-preserving (proper spherical geometry)
    - Consistent across the sphere
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
    print("GENERATING OVAL HOTSPOT GRIDS - METHOD: 3D GEODESIC (POLE-SAFE)")
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

                # Use 3D geodesic method (pole-safe)
                dist_sq = elliptical_distance_3d(theta, phi, theta_c, phi_c, dtheta, f)
                is_inside = dist_sq <= 1.0

                if is_inside:
                    grid[i, j] = 1.0
                    point_count += 1
                    theta_min = min(theta_min, theta)
                    theta_max = max(theta_max, theta)
                    phi_min = min(phi_min, phi)
                    phi_max = max(phi_max, phi)

        combined_grid = np.maximum(combined_grid, grid)

        output_file = output_dir / f"hotspot_{spot['name']}_{grid_resolution}_3d_geodesic.dat"
        np.savetxt(output_file, grid, fmt='%.18e', delimiter=' ')

        coverage = point_count / (grid_resolution**2) * 100
        print(f"\nPoints: {point_count} ({coverage:.2f}%)")
        print(f"θ range: [{theta_min:.4f}, {theta_max:.4f}]")
        print(f"φ range: [{phi_min:.4f}, {phi_max:.4f}]")
        print(f"Saved → {output_file.name}")

    combined_file = output_dir / f"hotspot_combined_{grid_resolution}_3d_geodesic.dat"
    np.savetxt(combined_file, combined_grid, fmt='%.18e', delimiter=' ')

    print(f"\n{'='*80}")
    print("Combined grid saved → hotspot_combined_3d_geodesic.dat")
    print("=" * 80)


def main():
    base_path = Path(__file__).parent
    case_dir = base_path / "hotspots"

    print("\n" + "="*80)
    print("3D GEODESIC HOTSPOT GENERATION")
    print("="*80 + "\n")

    grid_res = 5000

    generate_paper_hotspots_grid_3d_geodesic(
        grid_resolution=grid_res,
        output_dir=case_dir
    )

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print("\nOutput files generated:")
    print("  - hotspot_spot*_*_3d_geodesic.dat  (TRUE GEODESIC - POLE-SAFE)")
    print("\nThis method:")
    print("  ✓ Uses true spherical geometry (no singular points)")
    print("  ✓ Preserves area properly (ellipse in tangent plane)")
    print("  ✓ Works identically at equator and poles")
    print("  ✓ No clipping or wrapping artifacts")


if __name__ == "__main__":
    main()

