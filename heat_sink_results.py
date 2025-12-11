"""
Heat Sink Fin Results Module

Validation and visualization of thermal simulation results.

Features:
- Heat balance verification
- Temperature constraint check
- 2D temperature field visualization
- Transient evolution plots
- Mass reporting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Import geometry and solver
from heat_sink_geometry import (
    n_centers,
    center_nodes,
    structural_coords,
    areas,
    boundaries,
    N_fins, Q_total, h_conv, T_amb, T_max,
    compute_mass, compute_total_heatsink_mass,
    get_cv_coordinates,
    blocks,
    nHeight, nWidth
)

from heat_sink_solver import solve_fin, HeatSinkSolver

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_temperature_field(T, title="Temperature Distribution", ax=None, show_colorbar=True):
    """
    Plot 2D temperature field on the fin cross-section.

    Parameters
    ----------
    T : ndarray
        Temperature at each CV [C]
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_colorbar : bool
        Whether to show colorbar

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    coords = get_cv_coordinates()

    # Create patches for each control volume
    patches = []
    colors = []

    for i, center in enumerate(center_nodes):
        # Get block nodes for this CV
        block = blocks[center]

        # Get coordinates of block corners (outer 4 corners)
        center_row = center // nWidth
        center_col = center % nWidth

        # Determine corner indices
        min_row = max(0, center_row - 1)
        max_row = min(nHeight - 1, center_row + 1)
        min_col = max(0, center_col - 1)
        max_col = min(nWidth - 1, center_col + 1)

        corners_idx = [
            min_row * nWidth + min_col,  # bottom-left
            min_row * nWidth + max_col,  # bottom-right
            max_row * nWidth + max_col,  # top-right
            max_row * nWidth + min_col   # top-left
        ]

        corners = structural_coords[corners_idx]
        polygon = Polygon(corners, closed=True)
        patches.append(polygon)
        colors.append(T[i])

    # Create collection
    collection = PatchCollection(patches, cmap='hot', edgecolor='gray', linewidth=0.5)
    collection.set_array(np.array(colors))
    collection.set_clim(T_amb, max(T_max, np.max(T)))

    ax.add_collection(collection)

    # Plot CV centers
    ax.scatter(coords[:, 0] * 1000, coords[:, 1] * 1000, c='black', s=10, zorder=5)

    # Mark base nodes
    base_coords = coords[boundaries['base']]
    ax.scatter(base_coords[:, 0] * 1000, base_coords[:, 1] * 1000, c='red', s=30,
               marker='^', label='Base (heat input)', zorder=6)

    # Mark tip nodes
    tip_coords = coords[boundaries['tip']]
    ax.scatter(tip_coords[:, 0] * 1000, tip_coords[:, 1] * 1000, c='blue', s=30,
               marker='v', label='Tip', zorder=6)

    # Scale axes to mm
    ax.set_xlim(coords[:, 0].min() * 1000 - 2, coords[:, 0].max() * 1000 + 2)
    ax.set_ylim(coords[:, 1].min() * 1000 - 2, coords[:, 1].max() * 1000 + 2)

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if show_colorbar:
        cbar = fig.colorbar(collection, ax=ax, label='Temperature [C]')
        # Mark T_max constraint
        cbar.ax.axhline(y=T_max, color='red', linestyle='--', linewidth=2)
        cbar.ax.text(1.5, T_max, f'T_max={T_max}C', va='center', color='red')

    return fig, ax


def plot_temperature_scatter(T, title="Temperature Distribution (Scatter)", ax=None):
    """
    Plot temperature field using scatter plot with color mapping.

    Parameters
    ----------
    T : ndarray
        Temperature at each CV [C]
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    coords = get_cv_coordinates()

    # Scatter plot with temperature colors
    scatter = ax.scatter(coords[:, 0] * 1000, coords[:, 1] * 1000, c=T,
                         cmap='hot', s=200, edgecolor='black', linewidth=0.5,
                         vmin=T_amb, vmax=max(T_max, np.max(T)))

    # Add temperature labels
    for i, (x, y) in enumerate(coords):
        ax.annotate(f'{T[i]:.1f}', (x * 1000, y * 1000), fontsize=6,
                   ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax, label='Temperature [C]')

    return fig, ax


def plot_transient_evolution(results, node_indices=None, ax=None):
    """
    Plot temperature vs time for selected nodes.

    Parameters
    ----------
    results : dict
        Results from solver.get_results()
    node_indices : list, optional
        Indices of nodes to plot. If None, auto-selects base, mid, tip.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    T_history = np.array(results['T_history'])
    time_history = np.array(results['time_history'])

    if node_indices is None:
        # Auto-select: one base, one middle, one tip
        base_idx = boundaries['base'][len(boundaries['base']) // 2]
        tip_idx = boundaries['tip'][len(boundaries['tip']) // 2]
        mid_idx = n_centers // 2
        node_indices = [base_idx, mid_idx, tip_idx]
        labels = ['Base', 'Middle', 'Tip']
    else:
        labels = [f'CV {i}' for i in node_indices]

    for idx, label in zip(node_indices, labels):
        ax.plot(time_history, T_history[:, idx], label=label, linewidth=2)

    ax.axhline(y=T_amb, color='gray', linestyle='--', label=f'T_amb = {T_amb} C')
    ax.axhline(y=T_max, color='red', linestyle='--', label=f'T_max = {T_max} C')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Temperature [C]')
    ax.set_title('Transient Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_convergence(results, ax=None):
    """
    Plot convergence history (max dT vs iteration).

    Parameters
    ----------
    results : dict
        Results from solver.get_results()
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    convergence = results['convergence_history']
    iterations = np.arange(1, len(convergence) + 1)

    ax.semilogy(iterations, convergence, 'b-', linewidth=1)
    ax.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max dT [C]')
    ax.set_title('Convergence History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_temperature_profile(T, direction='vertical', ax=None):
    """
    Plot temperature profile along vertical or horizontal line.

    Parameters
    ----------
    T : ndarray
        Temperature at each CV [C]
    direction : str
        'vertical' or 'horizontal'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    coords = get_cv_coordinates()

    if direction == 'vertical':
        # Find middle column
        x_mid = np.median(coords[:, 0])
        x_tolerance = (coords[:, 0].max() - coords[:, 0].min()) / 10
        mask = np.abs(coords[:, 0] - x_mid) < x_tolerance

        pos = coords[mask, 1] * 1000  # Y in mm
        temp = T[mask]
        label = 'Y [mm]'
        title = 'Vertical Temperature Profile (center column)'
    else:
        # Find middle row
        y_mid = np.median(coords[:, 1])
        y_tolerance = (coords[:, 1].max() - coords[:, 1].min()) / 10
        mask = np.abs(coords[:, 1] - y_mid) < y_tolerance

        pos = coords[mask, 0] * 1000  # X in mm
        temp = T[mask]
        label = 'X [mm]'
        title = 'Horizontal Temperature Profile (center row)'

    # Sort by position
    sort_idx = np.argsort(pos)
    pos = pos[sort_idx]
    temp = temp[sort_idx]

    ax.plot(pos, temp, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=T_max, color='red', linestyle='--', label=f'T_max = {T_max} C')
    ax.axhline(y=T_amb, color='gray', linestyle='--', label=f'T_amb = {T_amb} C')

    ax.set_xlabel(label)
    ax.set_ylabel('Temperature [C]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


# =============================================================================
# REPORT FUNCTIONS
# =============================================================================

def print_results_summary(results):
    """
    Print comprehensive results summary.

    Parameters
    ----------
    results : dict
        Results from solver.get_results()
    """
    print("\n" + "=" * 70)
    print("HEAT SINK FIN SIMULATION RESULTS")
    print("=" * 70)

    # Temperature
    print("\n[1] TEMPERATURE FIELD")
    print("-" * 40)
    print(f"    Maximum temperature:  {results['T_max']:.2f} C")
    print(f"    Minimum temperature:  {results['T_min']:.2f} C")
    print(f"    Average temperature:  {results['T_avg']:.2f} C")
    print(f"    Base average:         {results['T_base_avg']:.2f} C")
    print(f"    Tip average:          {results['T_tip_avg']:.2f} C")
    print(f"    Max location (mm):    ({results['max_location'][0]*1000:.2f}, {results['max_location'][1]*1000:.2f})")

    # Constraint
    print("\n[2] CONSTRAINT CHECK")
    print("-" * 40)
    if results['constraint_satisfied']:
        print(f"    [OK] SATISFIED: T_max = {results['T_max']:.2f} C < {T_max} C")
        margin = T_max - results['T_max']
        print(f"    Margin: {margin:.2f} C")
    else:
        print(f"    [X] VIOLATED: T_max = {results['T_max']:.2f} C >= {T_max} C")
        excess = results['T_max'] - T_max
        print(f"    Excess: {excess:.2f} C")

    # Heat balance
    hb = results['heat_balance']
    print("\n[3] HEAT BALANCE")
    print("-" * 40)
    print(f"    Heat input (Q_in):    {hb['Q_in']:.4f} W")
    print(f"    Heat convected:       {hb['Q_conv']:.4f} W")
    print(f"    Balance error:        {hb['Q_error']:.6f} W")
    print(f"    Error percentage:     {hb['error_percent']:.4f}%")

    if hb['error_percent'] < 1.0:
        print("    [OK] Heat balance verified (< 1% error)")
    else:
        print("    [!] Heat balance error > 1%")

    # Mass
    print("\n[4] MASS CALCULATION")
    print("-" * 40)
    single_mass = compute_mass() * 1000  # g
    total_mass = compute_total_heatsink_mass() * 1000  # g
    print(f"    Single fin mass:      {single_mass:.2f} g")
    print(f"    Number of fins:       {N_fins}")
    print(f"    Total heat sink mass: {total_mass:.2f} g")

    # Convergence
    print("\n[5] SOLVER PERFORMANCE")
    print("-" * 40)
    print(f"    Converged:            {'Yes' if results['converged'] else 'No'}")
    print(f"    Iterations:           {results['n_iterations']}")
    if len(results['time_history']) > 0:
        print(f"    Simulation time:      {results['time_history'][-1]:.4f} s")

    print("\n" + "=" * 70)


def generate_full_report(thickness_array=None, save_plots=True, show_plots=True):
    """
    Generate complete analysis report with plots.

    Parameters
    ----------
    thickness_array : ndarray, optional
        Custom thickness distribution
    save_plots : bool
        Save plots to files
    show_plots : bool
        Display plots

    Returns
    -------
    solver : HeatSinkSolver
        Solver instance
    results : dict
        Results dictionary
    """
    # Solve
    print("Running thermal simulation...")
    solver, results = solve_fin(thickness_array, verbose=True)

    # Print summary
    print_results_summary(results)

    # Create plots
    fig1, ax1 = plot_temperature_scatter(results['T'], "Steady-State Temperature Field")
    fig2, ax2 = plot_transient_evolution(results)
    fig3, ax3 = plot_convergence(results)
    fig4, ax4 = plot_temperature_profile(results['T'], direction='vertical')

    if save_plots:
        fig1.savefig('temperature_field.png', dpi=150, bbox_inches='tight')
        fig2.savefig('transient_evolution.png', dpi=150, bbox_inches='tight')
        fig3.savefig('convergence.png', dpi=150, bbox_inches='tight')
        fig4.savefig('temperature_profile.png', dpi=150, bbox_inches='tight')
        print("\nPlots saved to current directory.")

    if show_plots:
        plt.show()

    return solver, results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    solver, results = generate_full_report(save_plots=True, show_plots=True)
