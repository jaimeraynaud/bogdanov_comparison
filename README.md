# Bogdanov Hotspots Analysis

A Python toolkit for analyzing and visualizing hotspot data on neutron stars using spherical coordinate transformations and Mollweide projections.

## Overview

This project provides tools to:
- Transform 2D grid hotspot maps (2000×2000) to spherical coordinates (phi, theta)
- Generate reference hotspots from paper parameters
- Compare model-detected hotspots with paper reference data
- Visualize hotspots on Mollweide map projections

## Project Structure

```
polar_caps_visualization/
├── transform_data.py          # Core data transformation functions
├── visualization.py           # Plotting and visualization functions
└── README.md                  # This file
```

## Main Components

### `transform_data.py`

Core module for hotspot data transformation and analysis:

- **`angular_distance_on_sphere()`** - Calculate angular distance between points on a sphere using spherical law of cosines
- **`transform_hotspot_data()`** - Convert 2000×2000 grid to discretized spherical coordinates (phi, theta) with hotspot parameter calculation
- **`rotate_spherical_coordinates()`** - Rotate spherical coordinates using observer frame transformation
- **`generate_paper_hotspots()`** - Generate reference hotspot grid based on paper parameters
- **`spherical_to_grid()`** - Inverse operation: convert spherical coordinates back to 2000×2000 grid format
- **`test_hotspot_parameters()`** - Compare model and paper hotspot parameters with tolerance checks

### `visualization.py`

Plotting and visualization module:

- **`setup_mollweide_figure()`** - Create Mollweide projection figure with custom tick formatting (degrees)
- **`convert_phi_theta()`** - Convert spherical coordinates to Mollweide projection coordinates
- **`plot_best_fits()`** - Plot model hotspots (Wendy) and paper reference (Jaime) on Mollweide projection

## File Naming Convention

- **Wendy** - Model-detected hotspots (from grid data)
  - `spot1_spherical_coordinates_wendy.dat`
  - `spot2_spherical_coordinates_wendy.dat`
  
- **Jaime** - Paper reference hotspots
  - `spot1_spherical_coordinates_jaime.dat` (spherical coordinates)
  - `spot2_spherical_coordinates_jaime.dat` (spherical coordinates)
  - `test_case2_spot1_shift_highres2k_jaime.dat` (2000×2000 grid)
  - `test_case2_spot2_shift_highres2k_jaime.dat` (2000×2000 grid)

## Input Files Required

The transformation scripts expect:
- `test_case2_spot1_shift_highres2k.dat` - 2000×2000 grid for Spot 1
- `test_case2_spot2_shift_highres2k.dat` - 2000×2000 grid for Spot 2

These should be in the same directory as the Python scripts.

## Key Parameters

### Paper Reference Hotspots (from Bogdanov et al.)

**Spot 1:**
- Colatitude: θc1 = 0.6283 rad
- Azimuth: φ1 = 0.0 rad
- Angular radius: Δθ1 = 0.01 rad

**Spot 2:**
- Colatitude: θc2 = 2.077 rad
- Azimuth: φ2 = 3.5343 rad (0.5625 cycles)
- Angular radius: Δθ2 = 0.33 rad

Observer angle: θobs = 0.733 rad

## Coordinate Systems

### Spherical Coordinates Format
- **Phi (φ)**: Azimuthal angle, 0 to 2π radians
- **Theta (θ)**: Colatitude, 0 to π radians (0 = north pole, π = south pole)

### Grid Format (2000×2000)
- **Rows**: Theta dimension (colatitude from 0 to π)
- **Columns**: Phi dimension (azimuth from 0 to 2π)
- **Values**: 0 for non-hotspot, 1 for hotspot points

### Mollweide Projection
- X-axis: Longitude converted from phi (shifted by -π for plotting)
- Y-axis: Latitude converted from colatitude

## Dependencies

```
numpy
matplotlib
```

## Usage Example

```python
from transform_data import transform_hotspot_data, generate_paper_hotspots
from visualization import plot_best_fits

# Transform model data
model_params = transform_hotspot_data(
    'test_case2_spot1_shift_highres2k.dat',
    'spot1_spherical_coordinates_wendy.dat'
)

# Generate paper reference
paper_params = generate_paper_hotspots(
    grid_resolution=2000,
    output_dir='./'
)

# Visualize
plot_best_fits('./', pointSize=5)
```

## Output

The scripts generate:

1. **Spherical coordinate files** (2 columns: phi, theta)
   - Model data (Wendy)
   - Paper reference (Jaime)

2. **Grid format files** (2000×2000 with 0s and 1s)
   - Paper reference only (Jaime grid format)

3. **Visualization**
   - Mollweide projection with model and paper hotspots overlaid
   - Degree-based tick labels
   - Color-coded by spot and source (model vs. paper)

## License

MIT License

## References

Bogdanov et al. - Paper reference for hotspot parameters

