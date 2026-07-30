#!/usr/bin/env python3
"""Visualize j0030 oval hotspot grids in theta-phi coordinates."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


# ===============================
# NEW: 3-SPOT PAPER PARAMETERS
# ===============================

PAPER_PARAMS = {
    "j0030 Spot 1": {
        "theta_c": 2.330,
        "dtheta": 0.032,
        # "phi_c": 0.0,
        "phi_c": 2.61799,
        "f": 5.335,
    },
    "j0030 Spot 2": {
        "theta_c": 2.446,
        "dtheta": 0.029,
        "phi_c": 2.61799 - 2 * np.pi * 0.463,
        "f": 16.588,
    },
    "j0030 Spot 3": {
        "theta_c": 3.056,
        "dtheta": 0.087,
        "phi_c": 2.61799 - 2 * np.pi * 0.427,
        "f": 1.253,
    },
}


def compute_dphi(theta_c, dtheta, f):
    """Compute longitudinal half-width Δφ."""
    sin_tc = np.sin(theta_c)
    if sin_tc < 1e-8:
        return np.pi
    dphi = f * (dtheta / sin_tc)
    return min(dphi, np.pi)


def invert_phi(phi):
    """Invert phi direction while keeping values in [0, 2π)."""
    return (-phi) % (2.0 * np.pi)


def phi_theta_to_equal_area(phi, theta, center_latitude=0.0):
    """
    Convert spherical coordinates (phi, theta) to azimuthal equal-area projection.
    
    Args:
        phi: azimuthal angle [0, 2π] (longitude)
        theta: polar angle [0, π] (colatitude, 0=north pole, π/2=equator)
        center_latitude: latitude at center of projection (default 0=equator)
    
    Returns:
        x, y: projected coordinates
    """
    # Convert theta to latitude-like coordinate (-π/2 to π/2, where 0 is equator)
    latitude = np.pi / 2 - theta
    
    # Convert to radians from center
    lat_diff = latitude - center_latitude
    
    # Azimuthal Equal-Area projection
    # rho = 2 * sin((π/2 - lat_diff) / 2)
    rho = 2.0 * np.sin((np.pi / 2.0 - lat_diff) / 2.0)
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    return x, y


def grid_to_phi_theta(data, threshold=0.5):
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.shape}")

    n_phi = data.shape[0] - 1
    n_theta = data.shape[1] - 1

    hotspot_idx = np.argwhere(data > threshold)
    if hotspot_idx.size == 0:
        return np.array([]), np.array([])

    i = hotspot_idx[:, 0]
    j = hotspot_idx[:, 1]

    phi = i * (2.0 * np.pi / n_phi)
    cos_theta = -1.0 + j * (2.0 / n_theta)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return phi, theta


def load_spot_file(path):
    data = np.loadtxt(path)
    return grid_to_phi_theta(data)


def plot_j0030(directory, point_size=6.0, save_path=None, show=True):

    directory = Path(directory)

    spot_files = [
        directory / "hotspot_spot1.dat",
        directory / "hotspot_spot2.dat",
        directory / "hotspot_spot3.dat",
        ]

    titles = ["j0030 Spot 1", "j0030 Spot 2", "j0030 Spot 3"]
    colors = ["tab:red", "tab:green", "tab:blue"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    for spot_file, title, color in zip(spot_files, titles, colors):

        if not spot_file.exists():
            raise FileNotFoundError(f"Missing input file: {spot_file}")

        phi, theta = load_spot_file(spot_file)
        phi = invert_phi(phi)

        ax.scatter(
            phi,
            theta,
            s=point_size,
            color=color,
            alpha=0.9,
            edgecolors="none",
            label=f"{title} (n={phi.size})",
        )

        params = PAPER_PARAMS[title]

        theta_c = params["theta_c"]
        phi_c = invert_phi(params["phi_c"])
        dtheta = params["dtheta"]
        dphi = compute_dphi(theta_c, dtheta, params["f"])

        # Plot center
        ax.scatter(
            phi_c,
            theta_c,
            s=90,
            color="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )

        # Plot vertical extent (Δθ)
        ax.errorbar(
            phi_c,
            theta_c,
            yerr=dtheta,
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
        )

        # Plot horizontal extent (Δφ)
        ax.errorbar(
            phi_c,
            theta_c,
            xerr=dphi,
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
        )

        # Draw ellipse outline
        t = np.linspace(0, 2 * np.pi, 200)
        phi_ellipse = (phi_c + dphi * np.cos(t)) % (2.0 * np.pi)
        theta_ellipse = theta_c + dtheta * np.sin(t)

        ax.plot(phi_ellipse, theta_ellipse, color="black", linewidth=1.0)

    # Add observer angle line at theta = 1.012
    observer_theta = 1.012
    phi_observer = np.linspace(0, 2 * np.pi, 200)
    ax.axhline(y=observer_theta, color="k", linestyle="--", linewidth=1.5, label=f"Observer angle θ={observer_theta:.3f}")

    ax.set_title("j0030 Oval Hotspots (θ vs φ)")
    ax.set_xlabel("φ [rad]")
    ax.set_ylabel("θ [rad]")

    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(np.pi, 0.0)

    ax.xaxis.set_major_locator(MultipleLocator(np.pi / 4))
    ax.xaxis.set_minor_locator(MultipleLocator(np.pi / 8))
    ax.yaxis.set_major_locator(MultipleLocator(np.pi / 4))
    ax.yaxis.set_minor_locator(MultipleLocator(np.pi / 8))

    ax.grid(which="major", alpha=0.45)
    ax.grid(which="minor", alpha=0.25, linestyle="--")

    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig


def plot_j0030_mollweide(directory, point_size=6.0, save_path=None, show=True):
    """Plot hotspots in a Mollweide projection (lon/lat)."""
    directory = Path(directory)

    spot_files = [
        directory / "hotspot_spot1.dat",
        directory / "hotspot_spot2.dat",
        directory / "hotspot_spot3.dat",
    ]

    titles = ["j0030 Spot 1", "j0030 Spot 2", "j0030 Spot 3"]
    colors = ["tab:red", "tab:green", "tab:blue"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180, subplot_kw={"projection": "mollweide"})

    for spot_file, title, color in zip(spot_files, titles, colors):

        if not spot_file.exists():
            raise FileNotFoundError(f"Missing input file: {spot_file}")

        phi, theta = load_spot_file(spot_file)
        phi = invert_phi(phi)

        lon = (phi + np.pi) % (2 * np.pi) - np.pi
        lat = np.pi / 2 - theta

        ax.scatter(
            lon,
            lat,
            s=point_size,
            color=color,
            alpha=0.9,
            edgecolors="none",
            label=f"{title} (n={lon.size})",
        )

        params = PAPER_PARAMS[title]

        theta_c = params["theta_c"]
        phi_c = invert_phi(params["phi_c"])
        dtheta = params["dtheta"]
        dphi = compute_dphi(theta_c, dtheta, params["f"])

        lon_c = (phi_c + np.pi) % (2 * np.pi) - np.pi
        lat_c = np.pi / 2 - theta_c

        # Plot center
        ax.scatter(
            lon_c,
            lat_c,
            s=90,
            color="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )

        # Draw ellipse outline on sphere (proper spherical ellipse)
        # Sample with finer resolution to capture curvature
        t = np.linspace(0, 2 * np.pi, 500)
        
        # Proper spherical ellipse sampling
        phi_ellipse = (phi_c + dphi * np.cos(t)) % (2.0 * np.pi)
        theta_ellipse = theta_c + dtheta * np.sin(t)
        
        # Project to mollweide
        lon_ellipse = (phi_ellipse + np.pi) % (2 * np.pi) - np.pi
        lat_ellipse = np.pi / 2 - theta_ellipse

        ax.plot(lon_ellipse, lat_ellipse, color="black", linewidth=1.0)

    # Observer angle line at theta = 1.012 (constant latitude)
    observer_theta = 1.012
    lat_observer = np.pi / 2 - observer_theta
    lon_observer = np.linspace(-np.pi, np.pi, 300)
    ax.plot(
        lon_observer,
        np.full_like(lon_observer, lat_observer),
        color="k",
        linestyle="--",
        linewidth=1.5,
        label=f"Observer angle θ={observer_theta:.3f}",
    )

    ax.set_title("j0030 Oval Hotspots (Mollweide)")
    ax.set_xlabel("Longitude [rad]")
    ax.set_ylabel("Latitude [rad]")
    ax.grid(which="major", alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig


def plot_j0030_southpole(directory, point_size=6.0, save_path=None, show=True):
    """Plot a circular south-pole view using an orthographic projection."""
    directory = Path(directory)

    spot_files = [
        directory / "hotspot_spot1.dat",
        directory / "hotspot_spot2.dat",
        directory / "hotspot_spot3.dat",
    ]

    titles = ["j0030 Spot 1", "j0030 Spot 2", "j0030 Spot 3"]
    colors = ["tab:red", "tab:green", "tab:blue"]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)

    for spot_file, title, color in zip(spot_files, titles, colors):
        if not spot_file.exists():
            raise FileNotFoundError(f"Missing input file: {spot_file}")

        phi, theta = load_spot_file(spot_file)
        phi = invert_phi(phi)

        # Use same longitude convention as mollweide for consistency
        lon = (phi + np.pi) % (2 * np.pi) - np.pi

        # Orthographic projection from the south pole
        # Flip y-axis to match mollweide orientation while keeping clockwise direction
        r = np.sin(np.pi - theta)
        x = r * np.cos(lon)
        y = -r * np.sin(lon)

        ax.scatter(
            x,
            y,
            s=point_size,
            color=color,
            alpha=0.9,
            edgecolors="none",
            label=f"{title} (n={x.size})",
        )

        params = PAPER_PARAMS[title]
        theta_c = params["theta_c"]
        phi_c = invert_phi(params["phi_c"])
        dtheta = params["dtheta"]
        dphi = compute_dphi(theta_c, dtheta, params["f"])

        # Use same longitude convention as mollweide
        lon_c = (phi_c + np.pi) % (2 * np.pi) - np.pi

        r_c = np.sin(np.pi - theta_c)
        x_c = r_c * np.cos(lon_c)
        y_c = -r_c * np.sin(lon_c)

        ax.scatter(
            x_c,
            y_c,
            s=90,
            color="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )

        t = np.linspace(0, 2 * np.pi, 500)  # Finer sampling for better shape
        phi_ellipse = (phi_c + dphi * np.cos(t)) % (2.0 * np.pi)
        theta_ellipse = theta_c + dtheta * np.sin(t)
        lon_ellipse = (phi_ellipse + np.pi) % (2 * np.pi) - np.pi
        r_ellipse = np.sin(np.pi - theta_ellipse)
        x_ellipse = r_ellipse * np.cos(lon_ellipse)
        y_ellipse = -r_ellipse * np.sin(lon_ellipse)
        ax.plot(x_ellipse, y_ellipse, color="black", linewidth=1.0)

    # Draw latitude circles (constant theta)
    theta_circles = np.linspace(0, np.pi, 40)  # 9 circles from north to south pole
    for theta_val in theta_circles:
        r = np.sin(np.pi - theta_val)
        circle = np.linspace(0, 2 * np.pi, 200)
        x_circle = r * np.cos(circle)
        y_circle = r * np.sin(circle)
        ax.plot(x_circle, y_circle, color="gray", linewidth=0.5, alpha=0.4, linestyle="-")

    # Draw longitude meridians (constant phi)
    lon_meridians = np.linspace(-np.pi, np.pi, 16)  # 16 meridians in lon space
    for lon_val in lon_meridians:
        theta_line = np.linspace(0, np.pi, 100)
        r_line = np.sin(np.pi - theta_line)
        x_line = r_line * np.cos(lon_val)
        y_line = -r_line * np.sin(lon_val)
        ax.plot(x_line, y_line, color="gray", linewidth=0.5, alpha=0.4, linestyle="-")

    # Draw unit circle boundary
    boundary = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(boundary), np.sin(boundary), color="black", linewidth=1.0)

    ax.set_title("j0030 Oval Hotspots (South Pole View)")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig


def parse_args():
    script_dir = Path(__file__).parent
    default_dir = script_dir / "hotspots"

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=default_dir)
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    save_rect = args.save
    save_moll = None
    save_south = None
    if args.save:
        save_moll = args.save.with_name(f"{args.save.stem}_mollweide{args.save.suffix}")
        save_south = args.save.with_name(f"{args.save.stem}_southpole{args.save.suffix}")

    fig_rect = plot_j0030(
        directory=args.directory,
        point_size=args.point_size,
        save_path=save_rect,
        show=False,
    )
    fig_moll = plot_j0030_mollweide(
        directory=args.directory,
        point_size=args.point_size,
        save_path=save_moll,
        show=False,
    )
    fig_south = plot_j0030_southpole(
        directory=args.directory,
        point_size=args.point_size,
        save_path=save_south,
        show=False,
    )

    if args.no_show:
        plt.close(fig_rect)
        plt.close(fig_moll)
        plt.close(fig_south)
    else:
        plt.show()


if __name__ == "__main__":
    main()

