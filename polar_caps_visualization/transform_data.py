#!/usr/bin/env python3
"""
Transform hotspot data from 2000x2000 grid to spherical coordinates.

This script reads binary hotspot maps (2000x2000 grids) and converts them
to phi and cos(theta) coordinates, including only hotspot points (value = 1).
"""

import numpy as np
import sys
from pathlib import Path


def transform_hotspot_data(input_file, output_file):
    """
    Transform hotspot grid data to spherical coordinates with discretization.
    
    Parameters
    ----------
    input_file : str
        Path to input .dat file containing 2000x2000 grid of 0s and 1s
    output_file : str
        Path to output file with 2 columns: phi and theta (both discretized)
    
    Returns
    -------
    dict
        Dictionary containing 'theta_c', 'phi_c', 'radius' if hotspot found, else None
    """
    # Read the data file
    print(f"Reading {input_file}...")
    data = np.loadtxt(input_file)
    
    # Verify dimensions
    if data.shape != (2000, 2000):
        print(f"Warning: Expected shape (2000, 2000), got {data.shape}")
    
    n_theta, n_phi = data.shape
    n_steps = 2000  # Number of discretization steps
    
    # Create discretization bins
    # Phi: 0 to 2*pi in 2000 steps
    phi_bins = np.linspace(0, 2 * np.pi, n_steps + 1)
    phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
    
    # cos(theta): -1 to 1 in 2000 steps
    cos_theta_bins = np.linspace(-1, 1, n_steps + 1)
    cos_theta_centers = (cos_theta_bins[:-1] + cos_theta_bins[1:]) / 2
    
    phi_values = []
    theta_values = []
    
    for i_theta in range(n_theta):
        for i_phi in range(n_phi):
            # Check if this is a hotspot point
            # if data[i_theta, i_phi] == 1:
            if data[i_phi,i_theta] == 1:
                # Compute continuous spherical coordinates
                # phi goes from 0 to 2*pi
                phi = (i_phi / n_phi) * 2 * np.pi
                
                # theta goes from 0 to pi
                theta = (i_theta / n_theta) * np.pi
                cos_theta = np.cos(theta)
                
                # Discretize phi: find which bin it belongs to
                phi_bin = np.searchsorted(phi_bins, phi, side='right') - 1
                phi_bin = np.clip(phi_bin, 0, n_steps - 1)
                phi_discrete = phi_centers[phi_bin]
                
                # Discretize cos(theta): find which bin it belongs to
                cos_theta_bin = np.searchsorted(cos_theta_bins, cos_theta, side='right') - 1
                cos_theta_bin = np.clip(cos_theta_bin, 0, n_steps - 1)
                cos_theta_discrete = cos_theta_centers[cos_theta_bin]
                
                # Convert discrete cos(theta) back to theta
                theta_discrete = np.arccos(cos_theta_discrete)
                
                phi_values.append(phi_discrete)
                theta_values.append(theta_discrete)
    
    # Stack and save
    output_data = np.column_stack((phi_values, theta_values))
    
    n_hotspot_points = len(phi_values)
    print(f"Found {n_hotspot_points} hotspot points")
    
    # Calculate hotspot center and radius
    hotspot_params = None
    if n_hotspot_points > 0:
        # Convert to Cartesian to compute center
        x_vals = np.sin(theta_values) * np.cos(phi_values)
        y_vals = np.sin(theta_values) * np.sin(phi_values)
        z_vals = np.cos(theta_values)
        
        # Compute mean Cartesian coordinates
        x_mean = np.mean(x_vals)
        y_mean = np.mean(y_vals)
        z_mean = np.mean(z_vals)
        
        # Normalize to unit sphere (approximate center on sphere)
        norm = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
        if norm > 0:
            x_mean /= norm
            y_mean /= norm
            z_mean /= norm
        
        # Convert back to spherical coordinates
        theta_center = np.arccos(np.clip(z_mean, -1.0, 1.0))
        phi_center = np.arctan2(y_mean, x_mean)
        
        # Wrap phi to [0, 2π]
        if phi_center < 0:
            phi_center += 2 * np.pi
        
        # Calculate approximate angular radius (RMS distance from center)
        distances = []
        for phi, theta in zip(phi_values, theta_values):
            dist = angular_distance_on_sphere(theta, phi, theta_center, phi_center)
            distances.append(dist)
        
        angular_radius = np.mean(distances)
        
        # Print hotspot information
        print(f"\nHotspot Information:")
        print(f"  Colatitude (θc):  {theta_center:.6f} rad")
        print(f"  Azimuth (φ):      {phi_center:.6f} rad")
        print(f"    (zero of azimuth defined by the plane containing the observer)")
        print(f"  Angular radius (Δθ): {angular_radius:.6f} rad")
        print()
        
        hotspot_params = {
            'theta_c': theta_center,
            'phi_c': phi_center,
            'radius': angular_radius
        }
    
    print(f"Writing to {output_file}...")
    np.savetxt(output_file, output_data, fmt='%.18e', delimiter=' ')
    
    print(f"Done! Output saved to {output_file}")
    print(f"Output shape: {output_data.shape}")
    
    return hotspot_params


def angular_distance_on_sphere(theta1, phi1, theta2, phi2):
    """
    Calculate the angular distance between two points on a sphere.
    
    Uses the haversine formula for accurate calculations.
    
    Parameters
    ----------
    theta1, phi1 : float
        Spherical coordinates of first point (colatitude and azimuth in radians)
    theta2, phi2 : float
        Spherical coordinates of second point (colatitude and azimuth in radians)
    
    Returns
    -------
    distance : float
        Angular distance in radians
    """
    # Using the spherical law of cosines (more stable than haversine for this purpose)
    cos_distance = (np.cos(theta1) * np.cos(theta2) + 
                    np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2))
    # Clip to avoid numerical errors with arccos
    cos_distance = np.clip(cos_distance, -1.0, 1.0)
    distance = np.arccos(cos_distance)
    return distance


def rotate_spherical_coordinates(theta, phi, theta_rot):
    """
    Rotate spherical coordinates by rotating the coordinate system.
    
    This transforms coordinates from the star's intrinsic frame to the 
    observer's viewing frame by rotating around the y-axis by angle theta_rot.
    
    Parameters
    ----------
    theta : float
        Colatitude in the original frame (radians)
    phi : float
        Azimuth in the original frame (radians)
    theta_rot : float
        Rotation angle (observer's colatitude) in radians
    
    Returns
    -------
    theta_new, phi_new : float
        Rotated colatitude and azimuth in the observer's frame
    """
    # Convert spherical to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Apply rotation around y-axis by theta_rot
    # Rotation matrix for y-axis rotation:
    # [cos(θ)   0  sin(θ)]
    # [  0      1    0   ]
    # [-sin(θ)  0  cos(θ)]
    x_new = x * np.cos(theta_rot) + z * np.sin(theta_rot)
    y_new = y
    z_new = -x * np.sin(theta_rot) + z * np.cos(theta_rot)
    
    # Convert back to spherical coordinates
    theta_new = np.arccos(np.clip(z_new, -1.0, 1.0))
    phi_new = np.arctan2(y_new, x_new)
    
    # Wrap phi to [0, 2π]
    if phi_new < 0:
        phi_new += 2 * np.pi
    
    return theta_new, phi_new


def generate_paper_hotspots(grid_resolution=2000, output_dir=None):
    """
    Generate hotspot coordinates based on the paper's reference parameters.
    
    This function creates a spherical grid and checks each point to determine
    if it falls within the defined circular hotspot regions from the paper.
    Uses the paper's hotspot parameters directly without observer frame transformation.
    
    Parameters
    ----------
    grid_resolution : int
        Resolution of the spherical grid (number of points along each dimension)
    output_dir : str or Path, optional
        Output directory for the generated files. If None, uses current directory.
    
    Returns
    -------
    dict
        Dictionary with spot names as keys, containing {'theta_c', 'phi_c', 'radius'} for each
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    
    # Paper parameters for the two spots
    spots = [
        {
            'name': 'spot1',
            'theta_c': 0.6283,      # colatitude θc1 in radians
            'phi_c': 0.0,           # azimuth φ1 in radians
            'radius': 0.01,         # angular radius Δθ1 in radians
        },
        {
            'name': 'spot2',
            'theta_c': 2.077,       # colatitude θc2 in radians
            'phi_c': 3.5343,        # azimuth φ2 in radians
            'radius': 0.33,         # angular radius Δθ2 in radians
        }
    ]
    
    # Generate spherical grid
    print("=" * 60)
    print("PAPER REFERENCE HOTSPOTS GENERATION (JAIME)")
    print("=" * 60)
    print(f"\nGenerating {grid_resolution}x{grid_resolution} spherical grid...")
    
    # Create discretization for phi and theta
    phi_edges = np.linspace(0, 2 * np.pi, grid_resolution + 1)
    theta_edges = np.linspace(0, np.pi, grid_resolution + 1)
    
    # Create grid of cell centers
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    
    paper_params = {}
    
    # Process each spot
    for spot in spots:
        print(f"\nProcessing {spot['name']}...")
        print(f"  Center: θc = {spot['theta_c']:.4f} rad, φc = {spot['phi_c']:.4f} rad")
        print(f"  Radius: {spot['radius']:.4f} rad")
        
        phi_points = []
        theta_points = []
        
        # Check each grid point
        point_count = 0
        for theta in theta_centers:
            for phi in phi_centers:
                # Calculate angular distance from this point to the spot center
                distance = angular_distance_on_sphere(theta, phi, spot['theta_c'], spot['phi_c'])
                
                # Include point if it's within the spot radius
                if distance <= spot['radius']:
                    phi_points.append(phi)
                    theta_points.append(theta)
                    point_count += 1
        
        # Calculate center and radius from the generated points
        if point_count > 0:
            # Convert to Cartesian to compute center
            x_vals = np.sin(theta_points) * np.cos(phi_points)
            y_vals = np.sin(theta_points) * np.sin(phi_points)
            z_vals = np.cos(theta_points)
            
            # Compute mean Cartesian coordinates
            x_mean = np.mean(x_vals)
            y_mean = np.mean(y_vals)
            z_mean = np.mean(z_vals)
            
            # Normalize to unit sphere
            norm = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
            if norm > 0:
                x_mean /= norm
                y_mean /= norm
                z_mean /= norm
            
            # Convert back to spherical coordinates
            theta_center_calc = np.arccos(np.clip(z_mean, -1.0, 1.0))
            phi_center_calc = np.arctan2(y_mean, x_mean)
            
            # Wrap phi to [0, 2π]
            if phi_center_calc < 0:
                phi_center_calc += 2 * np.pi
            
            # Calculate angular radius
            distances = []
            for phi, theta in zip(phi_points, theta_points):
                dist = angular_distance_on_sphere(theta, phi, theta_center_calc, phi_center_calc)
                distances.append(dist)
            
            angular_radius_calc = np.mean(distances)
            
            paper_params[spot['name']] = {
                'theta_c': theta_center_calc,
                'phi_c': phi_center_calc,
                'radius': angular_radius_calc
            }
        
        # Save to file with JAIME naming
        output_data = np.column_stack((phi_points, theta_points))
        output_file = output_dir / f"{spot['name']}_spherical_coordinates_jaime.dat"
        
        print(f"  Found {point_count} grid points within the spot")
        print(f"  Writing to {output_file.name}...")
        np.savetxt(str(output_file), output_data, fmt='%.18e', delimiter=' ')
        
    print("\n" + "=" * 60)
    print("Paper reference hotspots generation complete!")
    print("=" * 60)
    
    return paper_params


def spherical_to_grid(input_file, output_file, grid_resolution=2000):
    """
    Convert spherical coordinates back to a 2000x2000 grid format.
    
    This is the inverse operation of transform_hotspot_data. It takes spherical
    coordinates (phi, theta) and converts them back to a 2D grid where 1 represents
    hotspot points and 0 represents non-hotspot points.
    
    Parameters
    ----------
    input_file : str
        Path to input file with 2 columns: phi and theta (in radians)
    output_file : str
        Path to output .dat file (2000x2000 grid of 0s and 1s)
    grid_resolution : int
        Grid resolution (default: 2000x2000)
    
    Returns
    -------
    None
        Saves output grid to disk in scientific notation format
    """
    print(f"Reading {input_file}...")
    data = np.loadtxt(input_file)
    
    # Handle case where there's only one point
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Initialize grid with zeros
    grid = np.zeros((grid_resolution, grid_resolution), dtype=float)
    
    n_points = len(data)
    print(f"Found {n_points} hotspot points")
    
    # Convert each spherical coordinate back to grid indices
    for i, (phi, theta) in enumerate(data):
        # Convert phi and theta to grid indices
        # phi ranges from 0 to 2*pi -> maps to 0 to grid_resolution
        # theta ranges from 0 to pi -> maps to 0 to grid_resolution
        
        i_phi = int((phi / (2 * np.pi)) * grid_resolution)
        i_theta = int((theta / np.pi) * grid_resolution)
        
        # Clip to valid range to handle boundary cases
        i_phi = np.clip(i_phi, 0, grid_resolution - 1)
        i_theta = np.clip(i_theta, 0, grid_resolution - 1)
        
        # Set grid point to 1 (hotspot)
        grid[i_theta, i_phi] = 1.0
    
    # Print grid statistics
    num_hotspots = np.sum(grid)
    coverage = (num_hotspots / (grid_resolution ** 2)) * 100
    print(f"Grid has {int(num_hotspots)} hotspot points ({coverage:.2f}% coverage)")
    
    # Write grid to file in scientific notation format (matching test_case2 format)
    print(f"Writing to {output_file}...")
    np.savetxt(output_file, grid, fmt='%.18e', delimiter=' ')
    
    print(f"Done! Output saved to {output_file}")
    print(f"Output shape: {grid.shape}")
    print()


def test_hotspot_parameters(model_params_list, paper_params, tolerance_theta=0.01, tolerance_phi=0.01, tolerance_radius=0.05):
    """
    Compare model-detected hotspot parameters with paper reference parameters.
    
    Parameters
    ----------
    model_params_list : list of dict
        List of model hotspot parameters, one dict per spot
    paper_params : dict
        Dictionary of paper hotspot parameters with spot names as keys
    tolerance_theta : float
        Tolerance for colatitude comparison in radians
    tolerance_phi : float
        Tolerance for azimuth comparison in radians
    tolerance_radius : float
        Tolerance for angular radius comparison in radians
    """
    print("\n" + "=" * 70)
    print("HOTSPOT PARAMETER COMPARISON TEST")
    print("=" * 70)
    
    spot_names = ['spot1', 'spot2']
    all_match = True
    
    for idx, spot_name in enumerate(spot_names):
        if idx >= len(model_params_list):
            print(f"\nWarning: No model parameters for {spot_name}")
            continue
            
        if spot_name not in paper_params:
            print(f"\nWarning: No paper parameters for {spot_name}")
            continue
        
        model = model_params_list[idx]
        paper = paper_params[spot_name]
        
        print(f"\n{spot_name.upper()}")
        print("-" * 70)
        
        # Compare colatitude
        theta_diff = abs(model['theta_c'] - paper['theta_c'])
        theta_match = theta_diff <= tolerance_theta
        print(f"Colatitude (θc):")
        print(f"  Model:  {model['theta_c']:.6f} rad")
        print(f"  Paper:  {paper['theta_c']:.6f} rad")
        print(f"  Diff:   {theta_diff:.6f} rad  {'✓ PASS' if theta_match else '✗ FAIL'}")
        if not theta_match:
            all_match = False
        
        # Compare azimuth (need to handle wrap-around)
        phi_diff = abs(model['phi_c'] - paper['phi_c'])
        # Handle 2π wrap-around
        if phi_diff > np.pi:
            phi_diff = 2 * np.pi - phi_diff
        phi_match = phi_diff <= tolerance_phi
        print(f"\nAzimuth (φ):")
        print(f"  Model:  {model['phi_c']:.6f} rad")
        print(f"  Paper:  {paper['phi_c']:.6f} rad")
        print(f"  Diff:   {phi_diff:.6f} rad  {'✓ PASS' if phi_match else '✗ FAIL'}")
        if not phi_match:
            all_match = False
        
        # Compare angular radius
        radius_diff = abs(model['radius'] - paper['radius'])
        radius_match = radius_diff <= tolerance_radius
        print(f"\nAngular radius (Δθ):")
        print(f"  Model:  {model['radius']:.6f} rad")
        print(f"  Paper:  {paper['radius']:.6f} rad")
        print(f"  Diff:   {radius_diff:.6f} rad  {'✓ PASS' if radius_match else '✗ FAIL'}")
        if not radius_match:
            all_match = False
    
    print("\n" + "=" * 70)
    if all_match:
        print("RESULT: ✓ ALL TESTS PASSED - Model and Paper hotspots match!")
    else:
        print("RESULT: ✗ SOME TESTS FAILED - Model and Paper hotspots do not match!")
    print("=" * 70)
    
    return all_match


def main():
    # Define input and output file paths
    base_path = Path(__file__).parent
    
    # Process spot 1 - MODEL data with "WENDY" suffix
    input_file_1 = base_path / "test_case2_spot1_shift_highres2k.dat"
    output_file_1 = base_path / "spot1_spherical_coordinates_wendy.dat"
    
    # Process spot 2 - MODEL data with "WENDY" suffix
    input_file_2 = base_path / "test_case2_spot2_shift_highres2k.dat"
    output_file_2 = base_path / "spot2_spherical_coordinates_wendy.dat"
    
    # Transform both files
    print("=" * 60)
    print("HOTSPOT DATA TRANSFORMATION (MODEL DATA - WENDY)")
    print("=" * 60)
    print()
    
    model_params_1 = transform_hotspot_data(str(input_file_1), str(output_file_1))
    print()
    
    model_params_2 = transform_hotspot_data(str(input_file_2), str(output_file_2))
    
    print()
    print("=" * 60)
    print("Model data transformation complete!")
    print("=" * 60)
    
    # Generate paper reference hotspots (JAIME)
    print("\n")
    paper_params = generate_paper_hotspots(grid_resolution=2000, output_dir=base_path)
    
    # Convert Jaime's spherical coordinates back to grid format
    print("\n")
    print("=" * 60)
    print("CONVERTING JAIME SPHERICAL COORDINATES TO GRID FORMAT")
    print("=" * 60)
    print()
    
    jaime_spot1_spherical = base_path / "spot1_spherical_coordinates_jaime.dat"
    jaime_spot1_grid = base_path / "test_case2_spot1_shift_highres2k_jaime.dat"
    
    jaime_spot2_spherical = base_path / "spot2_spherical_coordinates_jaime.dat"
    jaime_spot2_grid = base_path / "test_case2_spot2_shift_highres2k_jaime.dat"
    
    print("Processing Spot 1 (Jaime)...")
    spherical_to_grid(str(jaime_spot1_spherical), str(jaime_spot1_grid))
    
    print("Processing Spot 2 (Jaime)...")
    spherical_to_grid(str(jaime_spot2_spherical), str(jaime_spot2_grid))
    
    print("=" * 60)
    print("Grid file conversion complete!")
    print("=" * 60)
    
    # Run comparison test
    print("\n")
    model_params_list = [model_params_1, model_params_2]
    test_hotspot_parameters(model_params_list, paper_params)


if __name__ == "__main__":
    main()

