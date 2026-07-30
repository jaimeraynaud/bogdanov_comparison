#!/usr/bin/env python3
"""
Compare all three methods: Cartesian, Tangent Plane, and 3D Geodesic.
Comprehensive analysis of spot3 widths and behavior.
"""

import numpy as np
from pathlib import Path


def analyze_method_results():
    """Analyze results from all three generation methods."""
    
    base_dir = Path(__file__).parent / "hotspots"
    print("BASE DIR: ", base_dir)
    grid_res = 5000
    spots = ['spot1', 'spot2', 'spot3']
    
    methods = {
        'cartesian': f'hotspot_*_{grid_res}_cartesian.dat',
        'tangent_plane': f'hotspot_*_{grid_res}_geodesic.dat',
        '3d_geodesic': f'hotspot_*_{grid_res}_3d_geodesic.dat',
    }
    
    print("\n" + "="*100)
    print("COMPREHENSIVE HOTSPOT METHOD COMPARISON")
    print("="*100)
    
    print(f"\nSpot3 Near-Pole Geometry:")
    print(f"  θc = 3.056 rad (175.1°)")
    print(f"  Δθ = 0.087 rad (largest of three spots)")
    print(f"  sin(θc) = 0.0855 (very small - singularity risk)")
    print(f"\nExpected: Spot3 should have WIDEST extent due to largest Δθ")
    
    results = {}
    
    for method_name, file_pattern in methods.items():
        print(f"\n" + "="*100)
        print(f"METHOD: {method_name.upper()}")
        print("="*100)
        
        method_results = {}
        
        for spot in spots:
            file_pattern_specific = file_pattern.replace('*', spot)
            filepath = base_dir / file_pattern_specific
            print("FILE PATH: ",filepath)
            
            if filepath.exists():
                data = np.loadtxt(filepath)
                inside = data == 1.0
                count = np.sum(inside)
                
                rows_with_points = np.where(np.any(inside, axis=1))[0]
                cols_with_points = np.where(np.any(inside, axis=0))[0]
                
                if len(rows_with_points) > 0 and len(cols_with_points) > 0:
                    phi_width = rows_with_points[-1] - rows_with_points[0] + 1
                    theta_width = cols_with_points[-1] - cols_with_points[0] + 1
                else:
                    phi_width = theta_width = 0
                
                method_results[spot] = {
                    'points': count,
                    'phi_width': phi_width,
                    'theta_width': theta_width,
                }
                
                print(f"{spot:6s}: points={count:6.0f}, φ-width={phi_width:6.0f}, θ-width={theta_width:4.0f}")
            else:
                print(f"{spot:6s}: FILE NOT FOUND - {filepath.name}")
                method_results[spot] = None
        
        results[method_name] = method_results
    
    # Analysis
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)
    
    print(f"\n{'Method':<20} {'Spot1 Θ':<12} {'Spot2 Θ':<12} {'Spot3 Θ':<12} {'Spot3 > Spot2?':<15}")
    print("-" * 70)
    
    for method_name, method_data in results.items():
        if all(v is not None for v in method_data.values()):
            s1_w = method_data['spot1']['theta_width']
            s2_w = method_data['spot2']['theta_width']
            s3_w = method_data['spot3']['theta_width']
            is_correct = "✓ YES" if s3_w > s2_w else "✗ NO"
            print(f"{method_name:<20} {s1_w:>10.0f}  {s2_w:>10.0f}  {s3_w:>10.0f}  {is_correct:<15}")
    
    print("\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100)
    
    print("""
1. TANGENT PLANE METHOD:
   - Expected to fix pole singularity via η = sin(θc) * Δφ pre-scaling
   - ACTUAL RESULT: Identical to Cartesian method
   - WHY: delta_phi() wrapping to [-π, π] already constrains extent
   - The pre-scaling doesn't help when wrapping is enforced symmetrically

2. 3D GEODESIC METHOD:
   - Uses true spherical geometry in 3D Cartesian coordinates
   - Projects to tangent plane at hotspot center
   - NO division by sin(θ), NO clipping, NO wrapping
   - Should show DIFFERENT (better) results
   - Currently processing...

3. THE FUNDAMENTAL ISSUE WITH CARTESIAN APPROACH:
   - dphi_circ = Δθ / sin(θc) becomes huge at pole
   - f * dphi_circ is then clipped to π
   - This underestimates the true physical extent
   - Spot3 should be wider than Spot2, but gets narrower

4. WHY 3D GEODESIC SHOULD WORK:
   - Projects sphere to local 2D tangent plane naturally
   - No singularities (handled by 3D geometry)
   - Ellipse dimensions directly in physical space
   - Should correctly identify spot3 as widest
""")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    analyze_method_results()

