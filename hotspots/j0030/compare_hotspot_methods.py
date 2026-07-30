#!/usr/bin/env python3
"""
Compare hotspot generation methods: Geodesic (pole-safe) vs Cartesian (old).

Analyze and visualize the differences, especially for spot3 near the pole.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_hotspot_file(filepath, name=""):
    """Load and analyze a hotspot grid file."""
    data = np.loadtxt(filepath)
    
    # Find points that are 1 (inside hotspot)
    inside = data == 1.0
    count = np.sum(inside)
    
    # Find extent in each direction
    rows_with_points = np.where(np.any(inside, axis=1))[0]
    cols_with_points = np.where(np.any(inside, axis=0))[0]
    
    if len(rows_with_points) > 0 and len(cols_with_points) > 0:
        phi_min_idx = rows_with_points[0]
        phi_max_idx = rows_with_points[-1]
        theta_min_idx = cols_with_points[0]
        theta_max_idx = cols_with_points[-1]
        
        phi_width = phi_max_idx - phi_min_idx + 1
        theta_width = theta_max_idx - theta_min_idx + 1
        aspect_ratio = phi_width / theta_width if theta_width > 0 else 0
    else:
        phi_width = theta_width = aspect_ratio = 0
    
    return {
        'name': name,
        'points': count,
        'coverage': count / (data.shape[0] * data.shape[1]) * 100,
        'phi_width': phi_width,
        'theta_width': theta_width,
        'aspect_ratio': aspect_ratio,
        'grid_size': data.shape,
    }


def main():
    base_dir = Path(__file__).parent / "hotspots"
    
    spots = ['spot1', 'spot2', 'spot3']
    grid_res = 5000
    
    print("\n" + "="*80)
    print("HOTSPOT GENERATION COMPARISON: GEODESIC vs CARTESIAN")
    print("="*80 + "\n")
    
    print(f"Grid Resolution: {grid_res} x {grid_res}\n")
    
    results = []
    
    for spot in spots:
        print(f"{'='*80}")
        print(f"{spot.upper()}")
        print(f"{'='*80}\n")
        
        # Load geodesic version
        geodesic_file = base_dir / f"hotspot_{spot}_{grid_res}_geodesic.dat"
        if geodesic_file.exists():
            geo_stats = analyze_hotspot_file(geodesic_file, f"{spot} (geodesic)")
            print(f"GEODESIC (pole-safe):")
            print(f"  Points inside: {geo_stats['points']:,}")
            print(f"  Coverage: {geo_stats['coverage']:.3f}%")
            print(f"  Φ width: {geo_stats['phi_width']:.0f} cells")
            print(f"  Θ width: {geo_stats['theta_width']:.0f} cells")
            print(f"  Aspect ratio (φ/θ): {geo_stats['aspect_ratio']:.3f}")
        else:
            print(f"GEODESIC: File not found")
            geo_stats = None
        
        # Load cartesian version
        cartesian_file = base_dir / f"hotspot_{spot}_{grid_res}_cartesian.dat"
        if cartesian_file.exists():
            cart_stats = analyze_hotspot_file(cartesian_file, f"{spot} (cartesian)")
            print(f"\nCARTESIAN (old method):")
            print(f"  Points inside: {cart_stats['points']:,}")
            print(f"  Coverage: {cart_stats['coverage']:.3f}%")
            print(f"  Φ width: {cart_stats['phi_width']:.0f} cells")
            print(f"  Θ width: {cart_stats['theta_width']:.0f} cells")
            print(f"  Aspect ratio (φ/θ): {cart_stats['aspect_ratio']:.3f}")
        else:
            print(f"CARTESIAN: File not found")
            cart_stats = None
        
        # Compare
        if geo_stats and cart_stats:
            phi_diff = geo_stats['phi_width'] - cart_stats['phi_width']
            theta_diff = geo_stats['theta_width'] - cart_stats['theta_width']
            print(f"\nDIFFERENCE (geodesic - cartesian):")
            print(f"  Φ width diff: {phi_diff:+.0f} cells ({phi_diff/cart_stats['phi_width']*100:+.1f}%)")
            print(f"  Θ width diff: {theta_diff:+.0f} cells ({theta_diff/cart_stats['theta_width']*100:+.1f}%)")
            
            results.append({
                'spot': spot,
                'geodesic': geo_stats,
                'cartesian': cart_stats,
            })
        
        print()
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80 + "\n")
    
    print("Expected behavior:")
    print("  Spot1 (Δθ=0.032): Should have moderate width")
    print("  Spot2 (Δθ=0.029): Should have slightly smaller width than spot1")
    print("  Spot3 (Δθ=0.087): Should have LARGEST width (largest Δθ)\n")
    
    if results:
        print("Width comparison (smaller Θ width = wider hotspot in Θ direction):\n")
        
        print("GEODESIC METHOD (recommended):")
        for r in results:
            print(f"  {r['spot']:6s}: Φ width = {r['geodesic']['phi_width']:4.0f}, "
                  f"Θ width = {r['geodesic']['theta_width']:4.0f}")
        
        print("\nCARTESIAN METHOD (has issues):")
        for r in results:
            print(f"  {r['spot']:6s}: Φ width = {r['cartesian']['phi_width']:4.0f}, "
                  f"Θ width = {r['cartesian']['theta_width']:4.0f}")
        
        # Check if spot3 is now wider
        print("\n" + "="*80)
        spot3_geo_width = results[2]['geodesic']['phi_width']
        spot3_cart_width = results[2]['cartesian']['phi_width']
        spot2_geo_width = results[1]['geodesic']['phi_width']
        spot2_cart_width = results[1]['cartesian']['phi_width']
        
        print("KEY FINDING:")
        if spot3_geo_width > spot2_geo_width:
            print(f"✓ GEODESIC: Spot3 ({spot3_geo_width:.0f}) > Spot2 ({spot2_geo_width:.0f}) ✓")
            print("  This is CORRECT - spot3 has largest Δθ!")
        else:
            print(f"✗ GEODESIC: Spot3 ({spot3_geo_width:.0f}) ≤ Spot2 ({spot2_geo_width:.0f})")
        
        if spot3_cart_width > spot2_cart_width:
            print(f"✓ CARTESIAN: Spot3 ({spot3_cart_width:.0f}) > Spot2 ({spot2_cart_width:.0f})")
        else:
            print(f"✗ CARTESIAN: Spot3 ({spot3_cart_width:.0f}) ≤ Spot2 ({spot2_cart_width:.0f})")
            print("  This is the POLE SINGULARITY BUG - spot3 should be widest!")
        
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
The tangent plane method (geodesic) fixes the pole singularity by:
  1. Pre-scaling zonal distances by sin(θc): η = sin(θc) * Δφ
  2. Avoiding 1/sin(θc) division that blows up at poles
  3. Using local tangent plane coordinates (ξ, η) where singularity is handled
  
This produces physically correct hotspot sizes across all latitudes, including
near the south pole (spot3 at θ=3.056 ≈ 174.8°).
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

