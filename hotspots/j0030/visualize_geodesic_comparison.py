#!/usr/bin/env python3
"""
Visualize the difference between geodesic and Cartesian hotspot methods.
Creates side-by-side plots showing the pole singularity issue and the fix.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


def load_and_downsample(filepath, factor=10):
    """Load hotspot file and downsample for faster visualization."""
    data = np.loadtxt(filepath)
    return data[::factor, ::factor]


def visualize_methods():
    """Create comparison visualization."""
    base_dir = Path(__file__).parent / "hotspots"
    grid_res = 5000
    ds_factor = 10
    
    spots = ['spot1', 'spot2', 'spot3']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Hotspot Generation: Geodesic vs Cartesian\n(Grid {grid_res}x{grid_res}, downsampled by {ds_factor})',
                 fontsize=14, fontweight='bold')
    
    for idx, spot in enumerate(spots):
        # GEODESIC
        geodesic_file = base_dir / f"hotspot_{spot}_{grid_res}_geodesic.dat"
        if geodesic_file.exists():
            geo_data = load_and_downsample(geodesic_file, ds_factor)
            ax = axes[0, idx]
            im = ax.imshow(geo_data.T, cmap='viridis', origin='lower', aspect='auto')
            ax.set_title(f'{spot.upper()} - GEODESIC (pole-safe)', fontweight='bold')
            ax.set_xlabel('φ')
            ax.set_ylabel('θ')
            plt.colorbar(im, ax=ax, label='Inside')
        
        # CARTESIAN
        cartesian_file = base_dir / f"hotspot_{spot}_{grid_res}_cartesian.dat"
        if cartesian_file.exists():
            cart_data = load_and_downsample(cartesian_file, ds_factor)
            ax = axes[1, idx]
            im = ax.imshow(cart_data.T, cmap='viridis', origin='lower', aspect='auto')
            ax.set_title(f'{spot.upper()} - CARTESIAN (old)', fontweight='bold', color='red')
            ax.set_xlabel('φ')
            ax.set_ylabel('θ')
            plt.colorbar(im, ax=ax, label='Inside')
    
    plt.tight_layout()
    output_file = base_dir / f"hotspot_comparison_geodesic_vs_cartesian.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")
    
    plt.show()


def create_pole_analysis_figure():
    """Create detailed analysis of the pole singularity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Pole Singularity Analysis: Why Cartesian Fails at θ ≈ π',
                 fontsize=14, fontweight='bold')
    
    # Plot 1: sin(theta) behavior
    theta = np.linspace(0, np.pi, 1000)
    ax = axes[0]
    ax.plot(theta, np.sin(theta), 'b-', linewidth=2, label='sin(θ)')
    ax.axvline(2.330, color='green', linestyle='--', label='Spot1 (θ=2.33)')
    ax.axvline(2.446, color='orange', linestyle='--', label='Spot2 (θ=2.45)')
    ax.axvline(3.056, color='red', linestyle='--', label='Spot3 (θ=3.06, near pole)')
    ax.axhline(0.1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('θ (radians)', fontsize=11)
    ax.set_ylabel('sin(θ)', fontsize=11)
    ax.set_title('sin(θ) Dependency: The Problem at Poles', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim([0, np.pi])
    
    # Plot 2: dphi_circ (the calculated longitudinal extent)
    ax = axes[1]
    dtheta_values = np.array([0.032, 0.029, 0.087])  # Δθ for each spot
    spot_names = ['Spot1', 'Spot2', 'Spot3']
    sin_values = np.array([np.sin(2.330), np.sin(2.446), np.sin(3.056)])
    f_values = np.array([5.335, 16.588, 1.253])
    
    dphi_circ = dtheta_values / sin_values
    dphi_scaled = f_values * dphi_circ
    dphi_clipped = np.minimum(dphi_scaled, np.pi)
    
    x_pos = np.arange(len(spot_names))
    width = 0.25
    
    ax.bar(x_pos - width, dphi_circ, width, label='dphi_circ (Δθ/sin(θ))',
           color='skyblue', edgecolor='black')
    ax.bar(x_pos, dphi_scaled, width, label='dphi (f * dphi_circ)',
           color='orange', edgecolor='black')
    ax.bar(x_pos + width, dphi_clipped, width, label='dphi (clipped to π)',
           color='red', edgecolor='black', alpha=0.7)
    
    ax.axhline(np.pi, color='red', linestyle='--', linewidth=2, label='π (clipping limit)')
    ax.set_ylabel('Longitudinal Extent (radians)', fontsize=11)
    ax.set_title('The Problem: Pole Causes Blow-Up & Clipping', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(spot_names)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    ax.text(2, np.pi + 0.3, 'Spot3 clipped to π!\nPhysical size lost!',
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    output_file = Path(__file__).parent / "hotspots" / "pole_singularity_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Analysis figure saved: {output_file}")
    
    plt.show()


def main():
    print("\nGenerating hotspot comparison visualizations...\n")
    
    try:
        visualize_methods()
    except Exception as e:
        print(f"Could not generate method comparison: {e}")
    
    try:
        create_pole_analysis_figure()
    except Exception as e:
        print(f"Could not generate pole analysis: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

