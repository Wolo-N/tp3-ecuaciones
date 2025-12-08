"""
2D Finite Volume Method Solver for Heat Sink Base Plate
========================================================

Solves the 2D heat conduction equation on the heat sink base plate:
    ρ·c·∂T/∂t = k·∇²T + q'''

**Boundary Conditions**:
- Bottom: Heat flux from processor (q = Q_total / A_processor)
- Top: Heat sink to fins (modeled as distributed heat sinks)
- Sides: Adiabatic (symmetry) or convective

**Features**:
- Uses Q9 shape functions for coordinate transformation (mapped grids)
- Supports both steady-state and transient (Forward Euler) solutions
- Integrates with 1D fin solver for coupled analysis

Based on grid_matrices_2d_with_centers.py
"""

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

# =============================================================================
# PARAMETERS - Heat Sink Base Plate
# =============================================================================

# Base plate geometry (50mm × 50mm processor)
plate_width = 0.050      # [m] - X direction
plate_depth = 0.050      # [m] - Y direction
plate_thickness = 0.003  # [m] - 3mm base plate thickness

# Grid resolution
nWidth = 11   # nodes in X direction (must be odd for center nodes)
nHeight = 11  # nodes in Y direction (must be odd for center nodes)

# Material properties - Aluminum
k = 205.0      # thermal conductivity [W/m-K]
rho = 2700.0   # density [kg/m³]
c = 902.0      # specific heat [J/kg-K]
h_conv = 300.0 # convection coefficient [W/m²-K]

# Boundary conditions
Q_processor = 500.0      # heat from processor [W]
T_inf = 45.0             # ambient temperature [°C]
T_initial = 45.0         # initial temperature [°C]

# Transient parameters
dt = 0.01                # time step [s]
t_final = 120.0          # simulation time [s]
save_interval = 1.0      # save every N seconds


# =============================================================================
# Q9 SHAPE FUNCTIONS AND GRID GENERATION
# =============================================================================

def q9_shape_functions(xi, eta):
    """Q9 (9-node quadrilateral) shape functions"""
    L1 = 0.5 * xi * (xi - 1.0)
    L2 = 1.0 - xi**2
    L3 = 0.5 * xi * (xi + 1.0)

    M1 = 0.5 * eta * (eta - 1.0)
    M2 = 1.0 - eta**2
    M3 = 0.5 * eta * (eta + 1.0)

    N = np.array([
        L1*M1,  # N1 (-1,-1)
        L3*M1,  # N2 ( 1,-1)
        L3*M3,  # N3 ( 1, 1)
        L1*M3,  # N4 (-1, 1)
        L2*M1,  # N5 ( 0,-1)
        L3*M2,  # N6 ( 1, 0)
        L2*M3,  # N7 ( 0, 1)
        L1*M2,  # N8 (-1, 0)
        L2*M2   # N9 ( 0, 0)
    ])
    return N


def q9_interpolate_points(ctrl_pts, natural_coords):
    """Interpolate physical coordinates using Q9 shape functions"""
    ctrl_pts = np.asarray(ctrl_pts, dtype=float).reshape(9, 2)
    natural_coords = np.asarray(natural_coords, dtype=float).reshape(-1, 2)
    Nmat = np.vstack([q9_shape_functions(xi, eta) for xi, eta in natural_coords])
    physical_coords = Nmat @ ctrl_pts
    return physical_coords


def create_rectangular_control_points(width, depth):
    """Create Q9 control points for rectangular domain"""
    w, d = width / 2, depth / 2
    ctrl = np.array([
        [-w, -d],    # N1 (-1,-1)
        [ w, -d],    # N2 ( 1,-1)
        [ w,  d],    # N3 ( 1, 1)
        [-w,  d],    # N4 (-1, 1)
        [ 0, -d],    # N5 ( 0,-1)
        [ w,  0],    # N6 ( 1, 0)
        [ 0,  d],    # N7 ( 0, 1)
        [-w,  0],    # N8 (-1, 0)
        [ 0,  0]     # N9 ( 0, 0)
    ])
    return ctrl


def heron_triangle_area(p1, p2, p3):
    """Calculate triangle area using Heron's formula"""
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c) / 2.0
    area_sq = s * (s - a) * (s - b) * (s - c)
    return sqrt(max(0, area_sq))


# =============================================================================
# 2D GRID BUILDER
# =============================================================================

class Grid2D:
    """2D structured grid for FVM analysis"""

    def __init__(self, width, depth, nWidth, nHeight, thickness=None):
        """
        Initialize 2D grid.

        Parameters
        ----------
        width : float
            Domain width (X direction) [m]
        depth : float
            Domain depth (Y direction) [m]
        nWidth : int
            Number of nodes in X direction
        nHeight : int
            Number of nodes in Y direction
        thickness : float, optional
            Plate thickness for volume calculation [m]
        """
        self.width = width
        self.depth = depth
        self.nWidth = nWidth
        self.nHeight = nHeight
        self.thickness = thickness if thickness else plate_thickness

        # Create control points for rectangular domain
        self.ctrl_pts = create_rectangular_control_points(width, depth)

        # Generate grid
        self._build_grid()

    def _build_grid(self):
        """Build all grid components"""
        # Natural coordinates
        xi_coords = np.linspace(-1, 1, self.nWidth)
        eta_coords = np.linspace(-1, 1, self.nHeight)

        Xi, Eta = np.meshgrid(xi_coords, eta_coords, indexing='ij')
        natural_coords = np.column_stack((Xi.ravel(), Eta.ravel()))

        # Physical coordinates
        self.nodes = q9_interpolate_points(self.ctrl_pts, natural_coords)
        self.nTotal = self.nWidth * self.nHeight

        # Grid spacing in natural coords
        self.dxi = xi_coords[1] - xi_coords[0] if self.nWidth > 1 else 0
        self.deta = eta_coords[1] - eta_coords[0] if self.nHeight > 1 else 0

        # Average physical spacing
        self.dx = self.width / (self.nWidth - 1)
        self.dy = self.depth / (self.nHeight - 1)

        # Build connectivity
        self._build_neighbors()

        # Compute areas and volumes
        self._compute_areas()
        self._compute_interface_lengths()

    def idx(self, i, j):
        """Convert (i,j) to linear index"""
        if 0 <= i < self.nWidth and 0 <= j < self.nHeight:
            return i * self.nHeight + j
        return None

    def ij(self, n):
        """Convert linear index to (i,j)"""
        i = n // self.nHeight
        j = n % self.nHeight
        return i, j

    def _build_neighbors(self):
        """Build neighbor connectivity: [W, E, S, N]"""
        self.neighbours = np.full((self.nTotal, 4), None, dtype=object)

        for i in range(self.nWidth):
            for j in range(self.nHeight):
                n = self.idx(i, j)
                self.neighbours[n, 0] = self.idx(i-1, j)  # W
                self.neighbours[n, 1] = self.idx(i+1, j)  # E
                self.neighbours[n, 2] = self.idx(i, j-1)  # S
                self.neighbours[n, 3] = self.idx(i, j+1)  # N

    def _compute_areas(self):
        """Compute control volume areas using surrounding nodes"""
        self.areas = np.zeros(self.nTotal)

        for n in range(self.nTotal):
            i, j = self.ij(n)

            # Control volume bounds
            # Interior nodes: half-way between neighbors
            # Boundary nodes: extend to domain edge

            # X bounds
            if i == 0:
                x_w = -self.width / 2
                x_e = (self.nodes[n, 0] + self.nodes[self.neighbours[n, 1], 0]) / 2 if self.neighbours[n, 1] is not None else self.width / 2
            elif i == self.nWidth - 1:
                x_w = (self.nodes[n, 0] + self.nodes[self.neighbours[n, 0], 0]) / 2 if self.neighbours[n, 0] is not None else -self.width / 2
                x_e = self.width / 2
            else:
                x_w = (self.nodes[n, 0] + self.nodes[self.neighbours[n, 0], 0]) / 2
                x_e = (self.nodes[n, 0] + self.nodes[self.neighbours[n, 1], 0]) / 2

            # Y bounds
            if j == 0:
                y_s = -self.depth / 2
                y_n = (self.nodes[n, 1] + self.nodes[self.neighbours[n, 3], 1]) / 2 if self.neighbours[n, 3] is not None else self.depth / 2
            elif j == self.nHeight - 1:
                y_s = (self.nodes[n, 1] + self.nodes[self.neighbours[n, 2], 1]) / 2 if self.neighbours[n, 2] is not None else -self.depth / 2
                y_n = self.depth / 2
            else:
                y_s = (self.nodes[n, 1] + self.nodes[self.neighbours[n, 2], 1]) / 2
                y_n = (self.nodes[n, 1] + self.nodes[self.neighbours[n, 3], 1]) / 2

            self.areas[n] = abs(x_e - x_w) * abs(y_n - y_s)

        # Control volumes (area × thickness)
        self.volumes = self.areas * self.thickness

        # Verify total area
        total_area = sum(self.areas)
        expected_area = self.width * self.depth
        if abs(total_area - expected_area) / expected_area > 0.01:
            print(f"Warning: Total CV area ({total_area*1e6:.1f} mm²) != domain area ({expected_area*1e6:.1f} mm²)")

    def _compute_interface_lengths(self):
        """Compute interface lengths for flux calculations: [L_w, L_e, L_s, L_n]"""
        self.interface_lengths = np.zeros((self.nTotal, 4))

        for n in range(self.nTotal):
            i, j = self.ij(n)

            # West and East interfaces (vertical lines)
            if self.neighbours[n, 2] is not None and self.neighbours[n, 3] is not None:
                dy = abs(self.nodes[self.neighbours[n, 3], 1] - self.nodes[self.neighbours[n, 2], 1]) / 2
            else:
                dy = self.dy

            # South and North interfaces (horizontal lines)
            if self.neighbours[n, 0] is not None and self.neighbours[n, 1] is not None:
                dx = abs(self.nodes[self.neighbours[n, 1], 0] - self.nodes[self.neighbours[n, 0], 0]) / 2
            else:
                dx = self.dx

            self.interface_lengths[n, 0] = dy  # L_w
            self.interface_lengths[n, 1] = dy  # L_e
            self.interface_lengths[n, 2] = dx  # L_s
            self.interface_lengths[n, 3] = dx  # L_n

    def get_boundary_nodes(self):
        """Get indices of boundary nodes"""
        west = [self.idx(0, j) for j in range(self.nHeight)]
        east = [self.idx(self.nWidth-1, j) for j in range(self.nHeight)]
        south = [self.idx(i, 0) for i in range(self.nWidth)]
        north = [self.idx(i, self.nHeight-1) for i in range(self.nWidth)]
        return {'west': west, 'east': east, 'south': south, 'north': north}

    def get_interior_nodes(self):
        """Get indices of interior nodes"""
        interior = []
        for i in range(1, self.nWidth-1):
            for j in range(1, self.nHeight-1):
                interior.append(self.idx(i, j))
        return interior


# =============================================================================
# 2D HEAT CONDUCTION SOLVER
# =============================================================================

class BasePlateSolver:
    """2D heat conduction solver for heat sink base plate"""

    def __init__(self, grid, k=205.0, rho=2700.0, c=902.0, h=300.0, T_inf=45.0):
        """
        Initialize solver.

        Parameters
        ----------
        grid : Grid2D
            Computational grid
        k : float
            Thermal conductivity [W/m-K]
        rho : float
            Density [kg/m³]
        c : float
            Specific heat [J/kg-K]
        h : float
            Convection coefficient [W/m²-K]
        T_inf : float
            Ambient temperature [°C]
        """
        self.grid = grid
        self.k = k
        self.rho = rho
        self.c = c
        self.h = h
        self.T_inf = T_inf

        # Heat sources/sinks
        self.Q_bottom = 0.0  # Heat flux from bottom [W/m²]
        self.fin_positions = []  # Positions where fins remove heat
        self.fin_heat_rates = []  # Heat removed by each fin [W]

    def set_processor_heat(self, Q_total):
        """Set heat input from processor"""
        A_processor = self.grid.width * self.grid.depth
        self.Q_bottom = Q_total / A_processor  # W/m²

    def add_fin_heat_sink(self, i, j, Q_fin):
        """Add a fin that removes heat at location (i,j)"""
        n = self.grid.idx(i, j)
        if n is not None:
            self.fin_positions.append(n)
            self.fin_heat_rates.append(Q_fin)

    def set_uniform_fins(self, n_fins_x, n_fins_y, Q_per_fin):
        """Set uniform distribution of fins"""
        self.fin_positions = []
        self.fin_heat_rates = []

        # Place fins at evenly spaced interior points
        x_positions = np.linspace(1, self.grid.nWidth-2, n_fins_x, dtype=int)
        y_positions = np.linspace(1, self.grid.nHeight-2, n_fins_y, dtype=int)

        for i in x_positions:
            for j in y_positions:
                self.add_fin_heat_sink(i, j, Q_per_fin)

    def assemble_steady_system(self):
        """Assemble system matrix for steady-state solution"""
        n = self.grid.nTotal
        M = np.zeros((n, n))
        b = np.zeros(n)

        boundary = self.grid.get_boundary_nodes()

        for node in range(n):
            i, j = self.grid.ij(node)
            A_cv = self.grid.areas[node]

            # Get distances to neighbors
            dx_w = abs(self.grid.nodes[node, 0] - self.grid.nodes[self.grid.neighbours[node, 0], 0]) if self.grid.neighbours[node, 0] is not None else self.grid.dx
            dx_e = abs(self.grid.nodes[self.grid.neighbours[node, 1], 0] - self.grid.nodes[node, 0]) if self.grid.neighbours[node, 1] is not None else self.grid.dx
            dy_s = abs(self.grid.nodes[node, 1] - self.grid.nodes[self.grid.neighbours[node, 2], 1]) if self.grid.neighbours[node, 2] is not None else self.grid.dy
            dy_n = abs(self.grid.nodes[self.grid.neighbours[node, 3], 1] - self.grid.nodes[node, 1]) if self.grid.neighbours[node, 3] is not None else self.grid.dy

            # Interface lengths
            L_w = self.grid.interface_lengths[node, 0] * self.grid.thickness
            L_e = self.grid.interface_lengths[node, 1] * self.grid.thickness
            L_s = self.grid.interface_lengths[node, 2] * self.grid.thickness
            L_n = self.grid.interface_lengths[node, 3] * self.grid.thickness

            # Conduction coefficients
            a_W = self.k * L_w / dx_w if self.grid.neighbours[node, 0] is not None else 0
            a_E = self.k * L_e / dx_e if self.grid.neighbours[node, 1] is not None else 0
            a_S = self.k * L_s / dy_s if self.grid.neighbours[node, 2] is not None else 0
            a_N = self.k * L_n / dy_n if self.grid.neighbours[node, 3] is not None else 0

            # Source term: heat from bottom (processor)
            Q_source = self.Q_bottom * A_cv

            # Heat sink from fins
            if node in self.fin_positions:
                idx = self.fin_positions.index(node)
                Q_source -= self.fin_heat_rates[idx]

            # Convection from top surface (exposed to air)
            h_top = self.h * A_cv

            # Boundary conditions
            if node in boundary['west']:
                a_W = 0  # Adiabatic
            if node in boundary['east']:
                a_E = 0  # Adiabatic
            if node in boundary['south']:
                a_S = 0  # Adiabatic
            if node in boundary['north']:
                a_N = 0  # Adiabatic

            # Assemble
            a_P = a_W + a_E + a_S + a_N + h_top

            M[node, node] = -a_P
            if self.grid.neighbours[node, 0] is not None:
                M[node, self.grid.neighbours[node, 0]] = a_W
            if self.grid.neighbours[node, 1] is not None:
                M[node, self.grid.neighbours[node, 1]] = a_E
            if self.grid.neighbours[node, 2] is not None:
                M[node, self.grid.neighbours[node, 2]] = a_S
            if self.grid.neighbours[node, 3] is not None:
                M[node, self.grid.neighbours[node, 3]] = a_N

            b[node] = -Q_source - h_top * self.T_inf

        return M, b

    def solve_steady_state(self):
        """Solve steady-state temperature distribution"""
        M, b = self.assemble_steady_system()
        T = np.linalg.solve(M, b)
        return T

    def solve_transient(self, T_init=None, dt=0.01, t_final=60.0, save_interval=1.0):
        """
        Solve transient using Forward Euler.

        Parameters
        ----------
        T_init : array, optional
            Initial temperature distribution. Default: T_inf everywhere
        dt : float
            Time step [s]
        t_final : float
            Final time [s]
        save_interval : float
            Interval to save results [s]

        Returns
        -------
        T_history : list
            Temperature distributions at saved times
        t_history : list
            Time values [s]
        T_final : array
            Final temperature distribution
        """
        n = self.grid.nTotal

        # Initial condition
        if T_init is None:
            T = np.ones(n) * self.T_inf
        else:
            T = T_init.copy()

        # Calculate stable time step
        alpha = self.k / (self.rho * self.c)
        dx_min = min(self.grid.dx, self.grid.dy)
        dt_stable = 0.25 * dx_min**2 / alpha

        if dt > dt_stable:
            print(f"Warning: dt={dt:.6f}s > stable dt={dt_stable:.6f}s. Reducing time step.")
            dt = 0.9 * dt_stable

        # Precompute thermal masses
        thermal_mass = self.rho * self.c * self.grid.volumes

        # Time stepping
        T_history = [T.copy()]
        t_history = [0.0]
        t = 0.0
        next_save = save_interval

        boundary = self.grid.get_boundary_nodes()
        n_steps = int(t_final / dt)

        for step in range(n_steps):
            T_new = T.copy()

            for node in range(n):
                i, j = self.grid.ij(node)
                A_cv = self.grid.areas[node]

                # Get distances
                dx_w = abs(self.grid.nodes[node, 0] - self.grid.nodes[self.grid.neighbours[node, 0], 0]) if self.grid.neighbours[node, 0] is not None else self.grid.dx
                dx_e = abs(self.grid.nodes[self.grid.neighbours[node, 1], 0] - self.grid.nodes[node, 0]) if self.grid.neighbours[node, 1] is not None else self.grid.dx
                dy_s = abs(self.grid.nodes[node, 1] - self.grid.nodes[self.grid.neighbours[node, 2], 1]) if self.grid.neighbours[node, 2] is not None else self.grid.dy
                dy_n = abs(self.grid.nodes[self.grid.neighbours[node, 3], 1] - self.grid.nodes[node, 1]) if self.grid.neighbours[node, 3] is not None else self.grid.dy

                # Interface areas (length × thickness)
                L_w = self.grid.interface_lengths[node, 0] * self.grid.thickness
                L_e = self.grid.interface_lengths[node, 1] * self.grid.thickness
                L_s = self.grid.interface_lengths[node, 2] * self.grid.thickness
                L_n = self.grid.interface_lengths[node, 3] * self.grid.thickness

                # Conduction fluxes
                Q_cond = 0.0
                if self.grid.neighbours[node, 0] is not None and node not in boundary['west']:
                    Q_cond += self.k * L_w * (T[self.grid.neighbours[node, 0]] - T[node]) / dx_w
                if self.grid.neighbours[node, 1] is not None and node not in boundary['east']:
                    Q_cond += self.k * L_e * (T[self.grid.neighbours[node, 1]] - T[node]) / dx_e
                if self.grid.neighbours[node, 2] is not None and node not in boundary['south']:
                    Q_cond += self.k * L_s * (T[self.grid.neighbours[node, 2]] - T[node]) / dy_s
                if self.grid.neighbours[node, 3] is not None and node not in boundary['north']:
                    Q_cond += self.k * L_n * (T[self.grid.neighbours[node, 3]] - T[node]) / dy_n

                # Source: heat from processor
                Q_source = self.Q_bottom * A_cv

                # Sink: fins
                if node in self.fin_positions:
                    idx = self.fin_positions.index(node)
                    Q_source -= self.fin_heat_rates[idx]

                # Convection from top surface
                Q_conv = self.h * A_cv * (self.T_inf - T[node])

                # Update temperature
                T_new[node] = T[node] + dt * (Q_cond + Q_source + Q_conv) / thermal_mass[node]

            T = T_new
            t += dt

            # Save at intervals
            if t >= next_save:
                T_history.append(T.copy())
                t_history.append(t)
                next_save += save_interval

        # Ensure final state is saved
        if t_history[-1] < t:
            T_history.append(T.copy())
            t_history.append(t)

        return T_history, t_history, T

    def compute_heat_balance(self, T):
        """
        Compute heat balance for verification.

        Returns
        -------
        Q_in : float
            Total heat input from processor [W]
        Q_conv : float
            Total heat convected from top surface [W]
        Q_fins : float
            Total heat removed by fins [W]
        error : float
            Relative balance error [%]
        """
        Q_in = self.Q_bottom * sum(self.grid.areas)
        Q_conv = sum(self.h * self.grid.areas[n] * (T[n] - self.T_inf) for n in range(self.grid.nTotal))
        Q_fins = sum(self.fin_heat_rates) if self.fin_heat_rates else 0.0

        Q_out = Q_conv + Q_fins
        error = abs(Q_in - Q_out) / Q_in * 100 if Q_in > 0 else 0.0

        return Q_in, Q_conv, Q_fins, error


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_temperature_field(grid, T, title="Temperature Distribution"):
    """Plot 2D temperature field"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reshape for contour plot
    T_2d = T.reshape(grid.nWidth, grid.nHeight)
    X = grid.nodes[:, 0].reshape(grid.nWidth, grid.nHeight) * 1000
    Y = grid.nodes[:, 1].reshape(grid.nWidth, grid.nHeight) * 1000

    # Handle uniform temperature case
    T_range = T.max() - T.min()
    if T_range < 0.01:
        levels = np.linspace(T.min() - 1, T.max() + 1, 20)
    else:
        levels = np.linspace(T.min(), T.max(), 20)

    # Contour plot
    cs = ax.contourf(X, Y, T_2d, levels=levels, cmap='hot')
    cbar = plt.colorbar(cs, ax=ax, label='Temperature [°C]')

    # Contour lines (only if there's variation)
    if T_range >= 0.01:
        ax.contour(X, Y, T_2d, levels=levels[::2], colors='white', linewidths=0.5, alpha=0.5)

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(title)
    ax.set_aspect('equal')

    return fig


def plot_transient_evolution(T_history, t_history, grid, locations=None):
    """Plot temperature evolution at specific locations"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if locations is None:
        # Default: center and corners
        n = grid.nTotal
        nw, nh = grid.nWidth, grid.nHeight
        locations = {
            'Center': grid.idx(nw//2, nh//2),
            'Corner (0,0)': grid.idx(0, 0),
            'Corner (W,0)': grid.idx(nw-1, 0),
            'Edge center': grid.idx(nw//2, 0)
        }

    for label, node in locations.items():
        T_at_loc = [T_history[i][node] for i in range(len(T_history))]
        ax.plot(t_history, T_at_loc, linewidth=1.5, label=f'{label} ({grid.nodes[node, 0]*1000:.1f}, {grid.nodes[node, 1]*1000:.1f}) mm')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title('Temperature Evolution at Different Locations')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("2D BASE PLATE SOLVER - HEAT SINK DESIGN (TP3)")
    print("="*60)

    # Create grid
    print(f"\nCreating {nWidth}×{nHeight} grid for {plate_width*1000:.0f}×{plate_depth*1000:.0f} mm plate...")
    grid = Grid2D(plate_width, plate_depth, nWidth, nHeight, plate_thickness)

    print(f"  Total nodes: {grid.nTotal}")
    print(f"  Grid spacing: dx={grid.dx*1000:.2f} mm, dy={grid.dy*1000:.2f} mm")
    print(f"  Plate thickness: {plate_thickness*1000:.1f} mm")
    print(f"  Total plate area: {sum(grid.areas)*1e6:.1f} mm²")
    print(f"  Total plate volume: {sum(grid.volumes)*1e9:.1f} mm³")

    # Create solver
    solver = BasePlateSolver(grid, k=k, rho=rho, c=c, h=h_conv, T_inf=T_inf)

    # Set heat input from processor
    solver.set_processor_heat(Q_processor)
    print(f"\nHeat input: {Q_processor} W ({solver.Q_bottom/1000:.1f} kW/m²)")

    # --- Case 1: No fins (just convection from top surface) ---
    print("\n>>> CASE 1: No fins (convection only) <<<")
    T_no_fins = solver.solve_steady_state()
    Q_in, Q_conv, Q_fins, error = solver.compute_heat_balance(T_no_fins)

    print(f"  Max temperature: {T_no_fins.max():.2f}°C")
    print(f"  Min temperature: {T_no_fins.min():.2f}°C")
    print(f"  Heat balance: Q_in={Q_in:.1f}W, Q_conv={Q_conv:.1f}W, error={error:.2f}%")

    fig1 = plot_temperature_field(grid, T_no_fins, "Temperature Distribution (No Fins)")
    fig1.savefig('base_plate_no_fins.png', dpi=150, bbox_inches='tight')
    print("  Saved: base_plate_no_fins.png")

    # --- Case 2: With fins ---
    print("\n>>> CASE 2: With fins <<<")
    # Estimate fins needed: if each fin removes ~25W (from 1D analysis)
    Q_per_fin = 25.0  # W per fin (approximate from 1D analysis)
    n_fins_total = int(np.ceil(Q_processor / Q_per_fin))
    n_fins_x = int(np.sqrt(n_fins_total))
    n_fins_y = int(np.ceil(n_fins_total / n_fins_x))

    print(f"  Estimated fins needed: {n_fins_total} ({n_fins_x}×{n_fins_y} grid)")

    # Reset solver with fins
    solver2 = BasePlateSolver(grid, k=k, rho=rho, c=c, h=h_conv, T_inf=T_inf)
    solver2.set_processor_heat(Q_processor)
    solver2.set_uniform_fins(n_fins_x, n_fins_y, Q_per_fin)

    T_with_fins = solver2.solve_steady_state()
    Q_in2, Q_conv2, Q_fins2, error2 = solver2.compute_heat_balance(T_with_fins)

    print(f"  Max temperature: {T_with_fins.max():.2f}°C")
    print(f"  Min temperature: {T_with_fins.min():.2f}°C")
    print(f"  Heat balance: Q_in={Q_in2:.1f}W, Q_conv={Q_conv2:.1f}W, Q_fins={Q_fins2:.1f}W, error={error2:.2f}%")

    fig2 = plot_temperature_field(grid, T_with_fins, f"Temperature Distribution ({n_fins_x}×{n_fins_y} Fins)")
    fig2.savefig('base_plate_with_fins.png', dpi=150, bbox_inches='tight')
    print("  Saved: base_plate_with_fins.png")

    # --- Transient analysis ---
    print(f"\n>>> TRANSIENT ANALYSIS <<<")
    print(f"Simulating from t=0 to t={t_final}s...")

    T_history, t_history, T_final = solver2.solve_transient(dt=dt, t_final=t_final, save_interval=save_interval)

    print(f"  Time steps saved: {len(t_history)}")
    print(f"  Final max temperature: {T_final.max():.2f}°C")
    print(f"  Difference from steady state: {np.max(np.abs(T_final - T_with_fins)):.4f}°C")

    fig3 = plot_transient_evolution(T_history, t_history, grid)
    fig3.savefig('base_plate_transient.png', dpi=150, bbox_inches='tight')
    print("  Saved: base_plate_transient.png")

    print("\n" + "="*60)
    print("BASE PLATE ANALYSIS COMPLETE")
    print("="*60)

    plt.show()
