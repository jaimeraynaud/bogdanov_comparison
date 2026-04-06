import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

# ------------------------------------------------------------------------------

def convert_phi_theta_3columns(data):
    mask = data[:, 2] == 1

    if not np.any(mask):
        return None

    phi = data[mask, 0] - np.pi
    # Changed as we are writing the cosine in fortran file as cth variable
    theta = np.pi / 2.0 - np.acos(data[mask, 1])
    #theta = np.pi / 2.0 - data[mask, 1]

    return phi, theta


def convert_phi_theta(data):
    # Convert phi from [0, 2π] to [-π, π]
    # phi = 0 stays at center (0)
    # phi = π stays at π (right side)
    # phi in (π, 2π] wraps to negative values
    phi = data[:, 0].copy()
    phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
    
    theta = np.pi/2.0 - data[:, 1]
    return phi, theta


def setup_mollweide_figure():
    fig = plt.figure(figsize=(8, 4), dpi=600)
    ax = fig.add_subplot(111, projection="mollweide")

    ax.set_facecolor('black')

    # Styling
    ax.tick_params(axis='both', colors='red', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_facecolor('black')
    # ax.set_yticks(np.arange(-np.pi / 2.0, np.pi / 2.0 + 0.01, np.pi / 8.0))
    # ax.set_xticks(np.arange(-np.pi, np.pi + 0.01, np.pi / 4.0))
    ax.set_yticks(np.arange(-np.pi / 2.0, np.pi / 2.0 + 0.01, np.pi / 16.0))
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.01, np.pi / 8.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{np.degrees(x):.1f}°"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{np.degrees(y):.1f}°"))
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True)

    return fig, ax


# ------------------------------------------------------------------------------
#  BEST-FIT PLOT (MODEL DATA + PAPER REFERENCE HOTSPOTS)
# ------------------------------------------------------------------------------

def plot_best_fits(directory_name, pointSize=0.5):
    print("Running BEST-FIT PLOT")

    # Model data files
    model_filenames = [
        "spot1_spherical_coordinates_wendy.dat", 
        "spot2_spherical_coordinates_wendy.dat"
    ]
    
    # Paper reference files
    paper_filenames = [
        "spot1_spherical_coordinates_jaime.dat",
        "spot2_spherical_coordinates_jaime.dat"
    ]

    # Labels and colors for model data (Wendy's)
    model_labels = ["Wendy's Spot 1", "Wendy's Spot 2"]
    model_colors = ["#FF8C00", "#FFB347"]  # dark orange and light orange
    model_alphas = [1.0, 1.0]
    
    # Labels and colors for paper reference data (Jaime's)
    paper_labels = ["Jaime's Spot 1", "Jaime's Spot 2"]
    paper_colors = ["#1E90FF", "#00CED1"]  # dodger blue and dark turquoise
    paper_alphas = [1.0, 1.0]

    fig, ax = setup_mollweide_figure()

    # Plot model data
    print("Plotting model hotspots...")
    for i, filename in enumerate(model_filenames):
        path = os.path.join(directory_name, filename)
        if os.path.exists(path):
            data = np.loadtxt(path)
            phi, theta = convert_phi_theta(data)
            ax.scatter(phi, theta, s=pointSize,
                        color=model_colors[i], alpha=model_alphas[i],
                        edgecolors='none', label=model_labels[i])
        else:
            print(f"Warning: {filename} not found")

    # Plot paper reference data
    print("Plotting paper reference hotspots...")
    for i, filename in enumerate(paper_filenames):
        path = os.path.join(directory_name, filename)
        if os.path.exists(path):
            data = np.loadtxt(path)
            phi, theta = convert_phi_theta(data)
            ax.scatter(phi, theta, s=pointSize,
                        color=paper_colors[i], alpha=paper_alphas[i],
                        label=paper_labels[i], marker='x')
        else:
            print(f"Warning: {filename} not found")

    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # plt.savefig("PC_StaticVSSweepback_best_l123.png", format='png', dpi=300)

    plt.show()

# ------------------------------------------------------------------------------
#  MAIN CALL
# ------------------------------------------------------------------------------

if __name__ == "__main__":

#For static versus sweepback plots
    pointSize = 5
    directory_name = "./"
    plot_best_fits(directory_name, pointSize)