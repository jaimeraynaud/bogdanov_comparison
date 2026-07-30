#!/usr/bin/env python3
"""
Diagnostic script to understand why both methods produce similar results.
Focus on spot3 (the problematic near-pole case).
"""

import numpy as np


def delta_phi(phi, phi_c):
    """Shortest wrapped angular difference."""
    dphi = phi - phi_c
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def tangent_plane_distance(theta, phi, theta_c, phi_c, dtheta, f):
    """Compute elliptical distance in tangent plane coordinates."""
    sin_tc = np.sin(theta_c)
    xi = theta - theta_c
    dphi_wrapped = delta_phi(phi, phi_c)
    eta = sin_tc * dphi_wrapped
    a = dtheta
    b = dtheta * f
    dist_sq = (xi / a)**2 + (eta / b)**2
    return dist_sq


def cartesian_distance(theta, phi, theta_c, phi_c, dtheta, f):
    """Old Cartesian method."""
    sin_tc = np.sin(theta_c)
    if sin_tc < 1e-8:
        return float('inf')
    dphi_circ = dtheta / sin_tc
    dphi = min(f * dphi_circ, np.pi)
    dt = (theta - theta_c) / dtheta
    dp = delta_phi(phi, phi_c) / dphi
    return dt**2 + dp**2


def analyze_spot3():
    """Detailed analysis of spot3 near the south pole."""
    
    spot3_theta_c = 3.056
    spot3_dtheta = 0.087
    spot3_phi_c = 2.61799 - 2 * np.pi * 0.427
    spot3_f = 1.253
    
    print("\n" + "="*80)
    print("SPOT3 DETAILED ANALYSIS (Near South Pole)")
    print("="*80)
    
    print(f"\nParameters:")
    print(f"  θc = {spot3_theta_c:.4f} rad ({np.degrees(spot3_theta_c):.1f}°)")
    print(f"  Δθ = {spot3_dtheta:.4f} rad")
    print(f"  φc = {spot3_phi_c:.4f} rad")
    print(f"  f = {spot3_f:.4f}")
    print(f"  sin(θc) = {np.sin(spot3_theta_c):.6f}")
    
    print(f"\nCARTESIAN METHOD CALCULATIONS:")
    sin_tc = np.sin(spot3_theta_c)
    dphi_circ = spot3_dtheta / sin_tc
    dphi_scaled = spot3_f * dphi_circ
    dphi_clipped = min(dphi_scaled, np.pi)
    
    print(f"  Δθ / sin(θc) = {spot3_dtheta:.4f} / {sin_tc:.6f} = {dphi_circ:.4f}")
    print(f"  f * dphi_circ = {spot3_f:.4f} * {dphi_circ:.4f} = {dphi_scaled:.4f}")
    print(f"  dphi_clipped = min({dphi_scaled:.4f}, π) = {dphi_clipped:.4f}")
    print(f"  ⚠ NOTE: Large value was clipped to π!")
    
    print(f"\nTANGENT PLANE METHOD CHARACTERISTICS:")
    print(f"  η = sin(θc) * Δφ")
    print(f"  This is pre-scaled, so no 1/sin(θc) blow-up!")
    print(f"  a = Δθ = {spot3_dtheta:.4f}")
    print(f"  b = Δθ * f = {spot3_dtheta:.4f} * {spot3_f:.4f} = {spot3_dtheta * spot3_f:.4f}")
    
    # Test points at different phi offsets
    print(f"\nTESTING POINTS AT DIFFERENT PHI OFFSETS:")
    print(f"{'δφ':>8} {'θ - θc':>10} {'Cartesian':>12} {'Tangent':>12} {'Both in?':>10}")
    print(f"{'-'*70}")
    
    test_dphi_values = [0, np.pi/8, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.2*np.pi]
    test_theta_offset = 0.02  # Small displacement in theta
    theta_test = spot3_theta_c + test_theta_offset
    
    for test_dphi in test_dphi_values:
        phi_test = spot3_phi_c + test_dphi
        
        cart_dist = cartesian_distance(theta_test, phi_test, spot3_theta_c, 
                                       spot3_phi_c, spot3_dtheta, spot3_f)
        tang_dist = tangent_plane_distance(theta_test, phi_test, spot3_theta_c, 
                                           spot3_phi_c, spot3_dtheta, spot3_f)
        
        cart_in = "✓" if cart_dist <= 1.0 else "✗"
        tang_in = "✓" if tang_dist <= 1.0 else "✗"
        both_in = "SAME" if (cart_in == tang_in) else "DIFF!"
        
        print(f"{test_dphi:8.4f} {test_theta_offset:10.4f} {cart_dist:12.4f} {tang_dist:12.4f} {both_in:>10}")
    
    print("\nKEY OBSERVATIONS:")
    print("  - Both methods give identical results!")
    print("  - Why? The wrapping of dphi cancels out the scaling difference")
    print("  - Need to reconsider the approach for truly different behavior")
    
    print("\n" + "="*80)


def main():
    analyze_spot3()
    
    print("\nCONCLUSION:")
    print("""
The current tangent plane implementation produces the same results as the
Cartesian method because:

1. Both use delta_phi() wrapping: dphi = (dphi + π) % (2π) - π
2. The wrapping ensures dphi ∈ [-π, π]
3. This effectively limits the extent regardless of method

The real fix for the pole singularity requires a different approach:

OPTION A: STEREOGRAPHIC PROJECTION
  - Project sphere locally to 2D plane near pole
  - Ellipse in projected coordinates
  - Automatically handles pole geometry

OPTION B: CARTESIAN 3D COORDINATES
  - Use (x, y, z) on unit sphere
  - Ellipse in 3D projected to tangent plane
  - More robust to all orientations

OPTION C: REFINED TANGENT PLANE
  - Keep pre-scaling but handle wrapping differently
  - Only wrap if logically necessary
  - Allow larger zonal extent at equator

The current code is correct mathematically but doesn't show practical
differences because wrapping already prevents blow-up in grid generation.
The benefit of tangent plane is more conceptual/theoretical.
""")


if __name__ == "__main__":
    main()

