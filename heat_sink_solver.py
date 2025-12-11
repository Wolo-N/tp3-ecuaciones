"""
Heat Sink Fin Solver Module

2D Transient Finite Volume Method (FVM) heat equation solver for a single fin.

FVM discretization:
    ρ·cp·V·(dT/dt) = Σ k·A_cond/dx·(T_nb - T) + h·A_conv·(T_amb - T) + Q_in

Where:
- A_cond: conduction area between CVs (depends on thickness & fin_depth)
- A_conv: convection area (exposed surfaces)
- Q_in: heat flux (only for base nodes)

Boundary conditions:
- Base (bottom): Heat flux Q_base = Q_total / N_fins
- All exposed surfaces: Convection h*(T_amb - T)
"""

import numpy as np
import matplotlib.pyplot as plt

# Import geometry module
from heat_sink_geometry import (
    n_centers,
    center_nodes,
    neighbours,
    areas,
    volumes,
    thickness,
    fin_depth,
    boundaries,
    rho, k, cp,
    N_fins, Q_total, h_conv, T_amb, T_max,
    compute_conduction_areas,
    compute_convection_areas,
    compute_neighbor_distances,
    get_cv_coordinates
)

# =============================================================================
# SOLVER PARAMETERS
# =============================================================================

# Time integration
dt_factor = 0.3     # Safety factor for time step (< 0.5 for stability)
max_iterations = 100000
convergence_tol = 1e-6  # Temperature change tolerance [°C]

# Heat input per fin
Q_per_fin = Q_total / N_fins  # [W]

# =============================================================================
# SOLVER CLASS
# =============================================================================

class HeatSinkSolver:
    """
    2D Transient FVM solver for heat sink fin thermal analysis.
    """

    def __init__(self, thickness_array=None):
        """
        Initialize solver with given geometry.

        Parameters
        ----------
        thickness_array : ndarray, optional
            Thickness per CV [m]. If None, uses default from geometry module.
        """
        self.n_cv = n_centers
        self.thickness = thickness_array if thickness_array is not None else thickness.copy()

        # Pre-compute geometry
        self.distances = compute_neighbor_distances()
        self.cond_areas = compute_conduction_areas(self.thickness)
        self.conv_areas = compute_convection_areas(self.thickness)

        # Initialize temperature field
        self.T = np.ones(self.n_cv) * T_amb

        # Build neighbor mapping (global to local indices)
        self.global_to_local = {node: idx for idx, node in enumerate(center_nodes)}

        # Compute stable time step
        self.dt = self._compute_stable_dt()

        # Storage for history
        self.T_history = []
        self.time_history = []
        self.convergence_history = []

    def _compute_stable_dt(self):
        """
        Compute stable time step for explicit Euler integration.

        Stability criterion (Fourier number):
            dt < (rho * cp * V) / (sum of all heat transfer coefficients)
        """
        dt_min = np.inf

        for i in range(self.n_cv):
            # Thermal mass
            thermal_mass = rho * cp * volumes[i]

            # Sum of conduction coefficients
            cond_coeff = 0.0
            for j in range(4):  # L, R, U, D
                if not np.isnan(self.distances[i, j]) and self.distances[i, j] > 0:
                    cond_coeff += k * self.cond_areas[i, j] / self.distances[i, j]

            # Convection coefficient
            conv_coeff = h_conv * self.conv_areas[i]

            # Total coefficient
            total_coeff = cond_coeff + conv_coeff

            if total_coeff > 0:
                dt_stable = thermal_mass / total_coeff
                dt_min = min(dt_min, dt_stable)

        return dt_factor * dt_min

    def _compute_heat_flux_base(self):
        """
        Compute heat input per base node.

        Returns
        -------
        Q_per_node : float
            Heat input per base node [W]
        """
        n_base_nodes = len(boundaries['base'])
        return Q_per_fin / n_base_nodes if n_base_nodes > 0 else 0.0

    def step(self):
        """
        Perform one explicit Euler time step.

        Returns
        -------
        dT_max : float
            Maximum temperature change this step
        """
        T_new = self.T.copy()
        Q_base_per_node = self._compute_heat_flux_base()

        for i in range(self.n_cv):
            # Thermal mass term
            thermal_mass = rho * cp * volumes[i]

            # Initialize heat rate
            dQ = 0.0

            # Conduction from neighbors
            nb = neighbours[i]  # [Center, Left, Right, Up, Down]
            for j, direction in enumerate([1, 2, 3, 4]):  # L, R, U, D
                if nb[direction] is not None:
                    neighbor_local = self.global_to_local.get(nb[direction])
                    if neighbor_local is not None and not np.isnan(self.distances[i, j]):
                        dx = self.distances[i, j]
                        A_cond = self.cond_areas[i, j]
                        dQ += k * A_cond / dx * (self.T[neighbor_local] - self.T[i])

            # Convection on exposed surfaces
            A_conv = self.conv_areas[i]
            dQ += h_conv * A_conv * (T_amb - self.T[i])

            # Heat input at base nodes
            if i in boundaries['base']:
                dQ += Q_base_per_node

            # Update temperature (explicit Euler)
            T_new[i] = self.T[i] + self.dt * dQ / thermal_mass

        # Compute max change
        dT_max = np.max(np.abs(T_new - self.T))

        # Update temperature
        self.T = T_new

        return dT_max

    def solve_steady_state(self, verbose=True, save_interval=100):
        """
        Iterate until steady state is reached.

        Parameters
        ----------
        verbose : bool
            Print progress information
        save_interval : int
            Save temperature history every N iterations

        Returns
        -------
        converged : bool
            Whether solution converged
        n_iterations : int
            Number of iterations performed
        """
        if verbose:
            print(f"\nSolving transient heat equation...")
            print(f"  Time step: {self.dt:.6e} s")
            print(f"  Convergence tolerance: {convergence_tol} C")

        self.T_history = [self.T.copy()]
        self.time_history = [0.0]
        self.convergence_history = []

        time = 0.0
        for iteration in range(max_iterations):
            dT_max = self.step()
            time += self.dt

            self.convergence_history.append(dT_max)

            if (iteration + 1) % save_interval == 0:
                self.T_history.append(self.T.copy())
                self.time_history.append(time)

                if verbose and (iteration + 1) % (10 * save_interval) == 0:
                    print(f"  Iteration {iteration + 1}: max dT = {dT_max:.6e} C, T_max = {np.max(self.T):.2f} C")

            if dT_max < convergence_tol:
                # Save final state
                self.T_history.append(self.T.copy())
                self.time_history.append(time)

                if verbose:
                    print(f"\n  Converged after {iteration + 1} iterations")
                    print(f"  Final time: {time:.4f} s")
                return True, iteration + 1

        if verbose:
            print(f"\n  WARNING: Did not converge after {max_iterations} iterations")
            print(f"  Final max dT = {dT_max:.6e} C")

        return False, max_iterations

    def compute_heat_balance(self):
        """
        Compute heat balance at steady state.

        Returns
        -------
        results : dict
            - Q_in: heat input at base [W]
            - Q_conv: heat convected to ambient [W]
            - Q_error: Q_in - Q_conv [W]
            - error_percent: percentage error
        """
        # Heat input
        Q_in = Q_per_fin

        # Heat convected (sum over all CVs)
        Q_conv = 0.0
        for i in range(self.n_cv):
            Q_conv += h_conv * self.conv_areas[i] * (self.T[i] - T_amb)

        Q_error = Q_in - Q_conv
        error_percent = 100 * abs(Q_error) / Q_in if Q_in > 0 else 0.0

        return {
            'Q_in': Q_in,
            'Q_conv': Q_conv,
            'Q_error': Q_error,
            'error_percent': error_percent
        }

    def get_results(self):
        """
        Get solver results summary.

        Returns
        -------
        results : dict
            Complete results dictionary
        """
        heat_balance = self.compute_heat_balance()

        coords = get_cv_coordinates()
        max_idx = np.argmax(self.T)

        return {
            'T': self.T.copy(),
            'T_max': np.max(self.T),
            'T_min': np.min(self.T),
            'T_avg': np.mean(self.T),
            'T_base_avg': np.mean(self.T[boundaries['base']]),
            'T_tip_avg': np.mean(self.T[boundaries['tip']]),
            'max_location': coords[max_idx],
            'max_idx': max_idx,
            'constraint_satisfied': np.max(self.T) < T_max,
            'heat_balance': heat_balance,
            'T_history': self.T_history,
            'time_history': self.time_history,
            'convergence_history': self.convergence_history
        }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_heat_sink_3d(T, title="Heat Sink - Single Fin 3D View", save_path=None):
    """
    Plot 3D visualization of ONE fin extruded in the Z direction.

    The fin cross-section (2D grid) is in the X-Y plane.
    The fin is extruded along Z by fin_depth (50mm).
    Heat enters from the base (bottom, min Y).

    Parameters
    ----------
    T : ndarray
        Temperature at each CV [C]
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig : matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from heat_sink_geometry import get_cv_coordinates, fin_depth, structural_coords, nWidth, nHeight, center_nodes

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    coords = get_cv_coordinates()

    # Color map
    cmap = plt.cm.hot
    T_min_plot, T_max_plot = T_amb, max(T_max, np.max(T))
    norm = plt.Normalize(T_min_plot, T_max_plot)

    # Draw ONE fin as extruded 3D shape
    # Each CV becomes a 3D box extruded in Z direction
    for i, center in enumerate(center_nodes):
        center_row = center // nWidth
        center_col = center % nWidth

        min_row = max(0, center_row - 1)
        max_row = min(nHeight - 1, center_row + 1)
        min_col = max(0, center_col - 1)
        max_col = min(nWidth - 1, center_col + 1)

        # Get corner coordinates in 2D (X-Y plane)
        corners_2d = [
            structural_coords[min_row * nWidth + min_col],
            structural_coords[min_row * nWidth + max_col],
            structural_coords[max_row * nWidth + max_col],
            structural_coords[max_row * nWidth + min_col]
        ]

        # Get color based on temperature
        color = cmap(norm(T[i]))

        # Create 3D box by extruding in Z direction
        z_front = 0.0
        z_back = fin_depth

        # Front face (Z = 0)
        verts_front = [[c[0], c[1], z_front] for c in corners_2d]

        # Back face (Z = fin_depth)
        verts_back = [[c[0], c[1], z_back] for c in corners_2d]

        # Draw front face
        poly_front = Poly3DCollection([verts_front], alpha=0.9)
        poly_front.set_facecolor(color)
        poly_front.set_edgecolor('gray')
        poly_front.set_linewidth(0.3)
        ax.add_collection3d(poly_front)

        # Draw back face
        poly_back = Poly3DCollection([verts_back], alpha=0.9)
        poly_back.set_facecolor(color)
        poly_back.set_edgecolor('gray')
        poly_back.set_linewidth(0.3)
        ax.add_collection3d(poly_back)

        # Draw side faces (connecting front and back)
        for j in range(4):
            j_next = (j + 1) % 4
            side_verts = [
                verts_front[j],
                verts_front[j_next],
                verts_back[j_next],
                verts_back[j]
            ]
            poly_side = Poly3DCollection([side_verts], alpha=0.7)
            poly_side.set_facecolor(color)
            poly_side.set_edgecolor('gray')
            poly_side.set_linewidth(0.2)
            ax.add_collection3d(poly_side)

    # Draw base plate (below the fin)
    base_thickness = 0.003  # 3mm base plate
    base_x = [coords[:, 0].min() - 0.005, coords[:, 0].max() + 0.005]
    base_y_bottom = coords[:, 1].min() - base_thickness
    base_y_top = coords[:, 1].min()

    # Base plate - top surface
    base_top = [
        [base_x[0], base_y_top, 0],
        [base_x[1], base_y_top, 0],
        [base_x[1], base_y_top, fin_depth],
        [base_x[0], base_y_top, fin_depth]
    ]
    poly_base = Poly3DCollection([base_top], alpha=0.5)
    poly_base.set_facecolor('gray')
    poly_base.set_edgecolor('black')
    ax.add_collection3d(poly_base)

    # Set labels
    ax.set_xlabel('X [m] - Fin Width')
    ax.set_ylabel('Y [m] - Fin Height')
    ax.set_zlabel('Z [m] - Fin Depth (50mm)')
    ax.set_title(title)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, label='Temperature [C]')

    # Set viewing angle
    ax.view_init(elev=20, azim=-45)

    # Set axis limits
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    z_range = fin_depth

    max_range = max(x_range, y_range, z_range) * 0.6

    mid_x = (coords[:, 0].max() + coords[:, 0].min()) / 2
    mid_y = (coords[:, 1].max() + coords[:, 1].min()) / 2
    mid_z = fin_depth / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 1.2)
    ax.set_zlim(0, fin_depth)

    # Add annotation for heat input
    ax.text(mid_x, coords[:, 1].min() - 0.01, mid_z, 'Heat Input (Q)',
            ha='center', fontsize=10, color='red')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_heat_sink_top_view(T, title="Single Fin - Top View (X-Z plane)", save_path=None):
    """
    Plot top view of ONE fin looking down from above.
    Shows the fin cross-section width (X) vs depth (Z).
    Temperature is uniform along Z (extrusion direction).

    Parameters
    ----------
    T : ndarray
        Temperature at each CV [C]
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from heat_sink_geometry import get_cv_coordinates, fin_depth

    fig, ax = plt.subplots(figsize=(12, 6))

    coords = get_cv_coordinates()

    # Color map
    cmap = plt.cm.hot
    T_min_plot, T_max_plot = T_amb, max(T_max, np.max(T))
    norm = plt.Normalize(T_min_plot, T_max_plot)

    # Get the fin profile at the tip (max Y) - showing the tapered shape
    # Group CVs by their X coordinate
    x_coords = coords[:, 0]
    unique_x = np.unique(np.round(x_coords, 6))

    # Draw fin as rectangles showing X extent at each row
    y_coords = coords[:, 1]
    unique_y = np.unique(np.round(y_coords, 6))

    for i, y_level in enumerate(unique_y):
        mask = np.abs(y_coords - y_level) < 1e-5
        T_avg_level = np.mean(T[mask])
        color = cmap(norm(T_avg_level))

        x_min, x_max = coords[mask, 0].min(), coords[mask, 0].max()
        x_width = x_max - x_min

        # Draw rectangle from Z=0 to Z=fin_depth at this X position
        rect = plt.Rectangle(
            (x_min * 1000, 0),
            x_width * 1000,
            fin_depth * 1000,
            facecolor=color,
            edgecolor='gray',
            linewidth=0.5
        )
        ax.add_patch(rect)

    # Labels
    ax.set_xlabel('X [mm] - Fin Width')
    ax.set_ylabel('Z [mm] - Fin Depth')
    ax.set_title(title)

    ax.set_xlim(coords[:, 0].min() * 1000 - 5, coords[:, 0].max() * 1000 + 5)
    ax.set_ylim(-2, fin_depth * 1000 + 2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Temperature [C]')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0, fin_depth * 1000 / 2, 'Uniform T\nalong depth',
            ha='center', va='center', fontsize=9, style='italic')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_heat_sink_side_view(T, title="Single Fin - Side View (X-Y cross-section)", save_path=None):
    """
    Plot side view showing temperature distribution on the fin cross-section (X-Y plane).

    Parameters
    ----------
    T : ndarray
        Temperature at each CV [C]
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from heat_sink_geometry import get_cv_coordinates, structural_coords, center_nodes, nWidth, nHeight

    fig, ax = plt.subplots(figsize=(10, 8))

    coords = get_cv_coordinates()

    # Color map
    cmap = plt.cm.hot
    T_min_plot, T_max_plot = T_amb, max(T_max, np.max(T))
    norm = plt.Normalize(T_min_plot, T_max_plot)

    patches = []
    colors = []

    for i, center in enumerate(center_nodes):
        center_row = center // nWidth
        center_col = center % nWidth

        min_row = max(0, center_row - 1)
        max_row = min(nHeight - 1, center_row + 1)
        min_col = max(0, center_col - 1)
        max_col = min(nWidth - 1, center_col + 1)

        corners_idx = [
            min_row * nWidth + min_col,
            min_row * nWidth + max_col,
            max_row * nWidth + max_col,
            max_row * nWidth + min_col
        ]

        corners = structural_coords[corners_idx] * 1000  # Convert to mm
        polygon = Polygon(corners, closed=True)
        patches.append(polygon)
        colors.append(T[i])

    collection = PatchCollection(patches, cmap=cmap, edgecolor='gray', linewidth=0.5)
    collection.set_array(np.array(colors))
    collection.set_clim(T_min_plot, T_max_plot)

    ax.add_collection(collection)

    # Mark base and tip
    base_coords = coords[boundaries['base']] * 1000
    tip_coords = coords[boundaries['tip']] * 1000

    ax.scatter(base_coords[:, 0], base_coords[:, 1], c='red', s=50,
               marker='^', label='Base (Q_in)', zorder=5)
    ax.scatter(tip_coords[:, 0], tip_coords[:, 1], c='blue', s=50,
               marker='v', label='Tip', zorder=5)

    # Add temperature annotations
    for i in range(len(coords)):
        ax.annotate(f'{T[i]:.1f}', (coords[i, 0] * 1000, coords[i, 1] * 1000),
                   fontsize=6, ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(title)

    ax.set_xlim(coords[:, 0].min() * 1000 - 3, coords[:, 0].max() * 1000 + 3)
    ax.set_ylim(coords[:, 1].min() * 1000 - 3, coords[:, 1].max() * 1000 + 3)

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(collection, ax=ax, label='Temperature [C]')
    cbar.ax.axhline(y=T_max, color='red', linestyle='--', linewidth=2)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_heat_sink_complete(T, save_prefix=None, show=True):
    """
    Generate complete visualization of the single fin with multiple views.

    Parameters
    ----------
    T : ndarray
        Temperature at each CV [C]
    save_prefix : str, optional
        Prefix for saving figures
    show : bool
        Whether to display plots

    Returns
    -------
    figs : dict
        Dictionary of figures
    """
    from heat_sink_geometry import fin_depth

    figs = {}

    # 3D view - single fin extruded in Z
    figs['3d'] = plot_heat_sink_3d(
        T,
        title=f"Single Fin 3D View (extruded {fin_depth*1000:.0f}mm in Z)\nT_max = {np.max(T):.1f} C",
        save_path=f"{save_prefix}_3d.png" if save_prefix else None
    )

    # Side view (X-Y cross-section)
    fig_side, _ = plot_heat_sink_side_view(
        T,
        title=f"Fin Cross-Section (X-Y plane)\nT_max = {np.max(T):.1f} C",
        save_path=f"{save_prefix}_side.png" if save_prefix else None
    )
    figs['side'] = fig_side

    # Top view (X-Z plane)
    fig_top, _ = plot_heat_sink_top_view(
        T,
        title=f"Fin Top View (X-Z plane)\nT_max = {np.max(T):.1f} C",
        save_path=f"{save_prefix}_top.png" if save_prefix else None
    )
    figs['top'] = fig_top

    if show:
        plt.show()

    return figs


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def solve_fin(thickness_array=None, verbose=True):
    """
    Solve heat equation for a fin with given thickness distribution.

    Parameters
    ----------
    thickness_array : ndarray, optional
        Thickness per CV [m]. If None, uses default.
    verbose : bool
        Print progress information

    Returns
    -------
    solver : HeatSinkSolver
        Solver instance with results
    results : dict
        Results dictionary
    """
    solver = HeatSinkSolver(thickness_array)
    converged, n_iter = solver.solve_steady_state(verbose=verbose)
    results = solver.get_results()
    results['converged'] = converged
    results['n_iterations'] = n_iter
    return solver, results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HEAT SINK FIN SOLVER")
    print("=" * 70)

    print(f"\nProblem Setup:")
    print(f"  - Number of CVs: {n_centers}")
    print(f"  - Heat input per fin: {Q_per_fin:.1f} W")
    print(f"  - Convection coefficient: {h_conv} W/m2-K")
    print(f"  - Ambient temperature: {T_amb} C")
    print(f"  - Max allowable temperature: {T_max} C")

    # Solve
    solver, results = solve_fin(verbose=True)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTemperature Field:")
    print(f"  - Maximum: {results['T_max']:.2f} C at CV {results['max_idx']}")
    print(f"  - Minimum: {results['T_min']:.2f} C")
    print(f"  - Average: {results['T_avg']:.2f} C")
    print(f"  - Base average: {results['T_base_avg']:.2f} C")
    print(f"  - Tip average: {results['T_tip_avg']:.2f} C")

    print(f"\nConstraint Check:")
    if results['constraint_satisfied']:
        print(f"  [OK] T_max ({results['T_max']:.2f} C) < {T_max} C - SATISFIED")
    else:
        print(f"  [X] T_max ({results['T_max']:.2f} C) >= {T_max} C - VIOLATED")

    hb = results['heat_balance']
    print(f"\nHeat Balance:")
    print(f"  - Heat input: {hb['Q_in']:.2f} W")
    print(f"  - Heat convected: {hb['Q_conv']:.2f} W")
    print(f"  - Error: {hb['Q_error']:.4f} W ({hb['error_percent']:.2f}%)")

    print("=" * 70)

    # Generate heat sink visualizations
    print("\nGenerating heat sink visualizations...")
    figs = plot_heat_sink_complete(results['T'], save_prefix='heat_sink', show=True)
