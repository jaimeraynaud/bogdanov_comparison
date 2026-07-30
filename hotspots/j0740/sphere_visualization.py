#!/usr/bin/env python3
"""Visualize J0740 hotspots on a 3D sphere and create a rotating GIF."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def grid_to_phi_theta(data, threshold=0.5):
    """Convert a hotspot grid into arrays of phi and theta hotspot coordinates."""
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {data.shape}")

    n_phi = data.shape[0] - 1
    n_theta = data.shape[1] - 1
    if n_phi <= 0 or n_theta <= 0:
        raise ValueError(f"Grid must be at least 2x2, got {data.shape}")

    hotspot_idx = np.argwhere(data > threshold)
    if hotspot_idx.size == 0:
        return np.array([]), np.array([])

    i = hotspot_idx[:, 0]
    j = hotspot_idx[:, 1]

    phi = i * (2.0 * np.pi / n_phi)
    cos_theta = -1.0 + j * (2.0 / n_theta)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return phi, theta


def load_spot_file(path, expected_shape=(5000, 5000)):
    """Load one J0740 spot grid and return phi/theta hotspot points."""
    data = np.loadtxt(path)
    if data.shape != expected_shape:
        print(f"Warning: Expected shape {expected_shape} for {path.name}, got {data.shape}")
    return grid_to_phi_theta(data)


def spherical_to_cartesian(phi, theta, radius=1.0):
    """Convert spherical (phi, theta) coordinates to Cartesian (x, y, z)."""
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z


def plot_sphere_with_hotspots(directory, point_size=6.0, hotspot_color="tab:red",
                              sphere_color="lightgray", save_gif=None,
                              save_static=None, n_frames=240, fps=20, show=True):
    """Plot hotspots on a 3D sphere and optionally save a rotating GIF."""
    directory = Path(directory).parent
    spot_files = [
        directory / "hotspot_spot1_5000_j0740.dat",
        directory / "hotspot_spot2_5000_j0740.dat",
        ]

    # Load both hotspots
    all_phi = []
    all_theta = []
    for spot_file in spot_files:
        if not spot_file.exists():
            raise FileNotFoundError(f"Missing input file: {spot_file}")
        phi, theta = load_spot_file(spot_file)
        all_phi.append(phi)
        all_theta.append(theta)

    # Convert hotspot points to Cartesian (slightly outside the sphere so they're visible)
    spot_radius = 1.01
    spots_xyz = []
    for phi, theta in zip(all_phi, all_theta):
        x, y, z = spherical_to_cartesian(phi, theta, radius=spot_radius)
        spots_xyz.append((x, y, z))

    # Build a sphere mesh
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    # Create figure
    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere surface
    ax.plot_surface(
        sphere_x, sphere_y, sphere_z,
        color=sphere_color, alpha=0.3,
        linewidth=0, antialiased=True, shade=True,
    )

    # Optional gridlines (meridians and parallels) for orientation
    for phi_line in np.linspace(0, 2 * np.pi, 13)[:-1]:
        theta_line = np.linspace(0, np.pi, 100)
        xL, yL, zL = spherical_to_cartesian(phi_line, theta_line)
        ax.plot(xL, yL, zL, color="gray", linewidth=0.3, alpha=0.4)
    for theta_line in np.linspace(0, np.pi, 7)[1:-1]:
        phi_line = np.linspace(0, 2 * np.pi, 100)
        xL, yL, zL = spherical_to_cartesian(phi_line, theta_line)
        ax.plot(xL, yL, zL, color="gray", linewidth=0.3, alpha=0.4)

    # Plot both hotspots in the same color (no label, no legend)
    for x, y, z in spots_xyz:
        ax.scatter(
            x, y, z,
            s=point_size,
            color=hotspot_color,
            alpha=0.9,
            edgecolors="none",
        )

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Save static image if requested
    if save_static is not None:
        fig.savefig(save_static, dpi=300, bbox_inches="tight")
        print(f"Saved static figure to {save_static}")

    # Create rotating GIF
    if save_gif is not None:
        def update(frame):
            azim = (frame / n_frames) * 360.0
            ax.view_init(elev=20, azim=azim)
            return []

        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
        writer = PillowWriter(fps=fps)
        anim.save(save_gif, writer=writer)
        print(f"Saved rotating GIF to {save_gif}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    script_dir = Path(__file__).parent
    default_dir = script_dir / "j0740"

    parser = argparse.ArgumentParser(
        description="Visualize J0740 hotspots on a 3D sphere and create a rotating GIF."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=default_dir,
        help=f"Directory containing J0740 .dat files (default: {default_dir})",
    )
    parser.add_argument("--point-size", type=float, default=6.0, help="Scatter point size.")
    parser.add_argument("--hotspot-color", type=str, default="tab:red",
                        help="Color used for both hotspots.")
    parser.add_argument("--sphere-color", type=str, default="lightgray",
                        help="Color of the sphere surface.")
    parser.add_argument("--save-gif", type=Path, default=Path("j0740_hotspots_rotating.gif"),
                        help="Output path for the rotating GIF (set to '' to skip).")
    parser.add_argument("--save-static", type=Path, default=None,
                        help="Optional output path for a static image.")
    parser.add_argument("--n-frames", type=int, default=240,
                        help="Number of frames in the rotating GIF (more = slower rotation).")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second for the GIF (lower = slower).")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    return parser.parse_args()


def main():
    args = parse_args()
    save_gif = args.save_gif if (args.save_gif and str(args.save_gif) != "") else None
    plot_sphere_with_hotspots(
        directory=args.directory,
        point_size=args.point_size,
        hotspot_color=args.hotspot_color,
        sphere_color=args.sphere_color,
        save_gif=save_gif,
        save_static=args.save_static,
        n_frames=args.n_frames,
        fps=args.fps,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()