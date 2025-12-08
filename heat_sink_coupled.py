"""
Coupled Heat Sink Solver
========================

Couples the 2D base plate solver with 1D fin analysis for complete heat sink simulation.

**Approach**:
1. Model base plate with 2D FVM (heat from processor, conduction, convection)
2. Model each fin with 1D FVM (given base temperature, calculates heat removed)
3. Iteratively couple: base temperature → fin heat removal → base temperature

**Features**:
- Parametric fin placement (number, spacing, geometry)
- Steady-state and transient coupled solutions
- Mass calculation for optimization
- Heat balance verification
"""

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import our solvers
from fin_fvm_1d import (solve_steady_state as solve_fin_steady,
                        solve_transient as solve_fin_transient,
                        calculate_fin_mass, print_results as print_fin_results,
                        A_fun, P_fun, build_grid as build_fin_grid,
                        calculate_heat_balance as calc_fin_heat_balance)
from base_plate_solver_2d import Grid2D, BasePlateSolver, plot_temperature_field

# =============================================================================
# PARAMETERS
# =============================================================================

# Base plate
PLATE_WIDTH = 0.050      # [m]
PLATE_DEPTH = 0.050      # [m]
PLATE_THICKNESS = 0.003  # [m]

# Fin geometry
FIN_HEIGHT = 0.025       # [m] - max 25mm
FIN_WIDTH = 0.050        # [m] - same as plate (50mm)
FIN_THICKNESS = 0.002    # [m] - 2mm

# Material - Aluminum
K = 205.0                # thermal conductivity [W/m-K]
RHO = 2700.0             # density [kg/m³]
C = 902.0                # specific heat [J/kg-K]
H = 300.0                # convection coefficient [W/m²-K]

# Boundary conditions
Q_PROCESSOR = 500.0      # [W]
T_INF = 45.0             # [°C]
T_MAX = 90.0             # [°C] max allowed

# Grid resolution
N_GRID = 21              # nodes in each direction for base plate


# =============================================================================
# COUPLED HEAT SINK MODEL
# =============================================================================

class CoupledHeatSink:
    """
    Coupled model of heat sink with base plate and fins.

    The model assumes:
    - Flat rectangular fins along the Y direction (perpendicular to X)
    - Fins are uniformly spaced along X
    - Each fin has uniform base temperature along its length
    """

    def __init__(self, n_fins, fin_height=FIN_HEIGHT, fin_thickness=FIN_THICKNESS,
                 plate_thickness=PLATE_THICKNESS):
        """
        Initialize coupled heat sink model.

        Parameters
        ----------
        n_fins : int
            Number of fins
        fin_height : float
            Fin height [m]
        fin_thickness : float
            Fin thickness [m]
        plate_thickness : float
            Base plate thickness [m]
        """
        self.n_fins = n_fins
        self.fin_height = fin_height
        self.fin_thickness = fin_thickness
        self.plate_thickness = plate_thickness

        # Calculate fin spacing
        self.fin_spacing = PLATE_WIDTH / n_fins if n_fins > 0 else PLATE_WIDTH

        # Create base plate grid (1D along X for fin bases, full 2D internally)
        self.grid = Grid2D(PLATE_WIDTH, PLATE_DEPTH, N_GRID, N_GRID, plate_thickness)

        # Create base plate solver
        self.plate_solver = BasePlateSolver(self.grid, k=K, rho=RHO, c=C, h=H, T_inf=T_INF)
        self.plate_solver.set_processor_heat(Q_PROCESSOR)

        # Fin positions (column indices where fins attach)
        self._calculate_fin_positions()

        # Store results
        self.T_base = None
        self.T_fins = []
        self.Q_fins = []
        self.converged = False

    def _calculate_fin_positions(self):
        """Calculate grid positions where fins attach"""
        self.fin_positions = []  # List of (i, j) grid positions for each fin

        if self.n_fins == 0:
            return

        # Fins are placed along X at regular intervals
        # Each fin spans the entire Y direction
        x_positions = np.linspace(0, self.grid.nWidth - 1, self.n_fins + 2)[1:-1]
        x_positions = np.round(x_positions).astype(int)

        for i_x in x_positions:
            # For each fin, it connects to all nodes along Y at this X position
            fin_nodes = []
            for j_y in range(self.grid.nHeight):
                fin_nodes.append((i_x, j_y))
            self.fin_positions.append(fin_nodes)

    def solve_single_fin(self, T_base):
        """
        Solve single fin given base temperature.

        Returns heat removed [W] and temperature profile.
        """
        T, x, Q_base, Q_conv, eta = solve_fin_steady(
            lFin_val=self.fin_height,
            T_base_val=T_base,
            T_inf_val=T_INF,
            k_val=K,
            h_val=H
        )
        return Q_conv, T, x, eta

    def solve_steady_coupled(self, max_iter=100, tol=0.1, relax=0.5, verbose=True):
        """
        Solve coupled steady-state problem iteratively with under-relaxation.

        Algorithm:
        1. Start with uniform base temperature guess
        2. Solve fins to get heat removed at each location
        3. Solve base plate with fin heat sinks
        4. Under-relax the update: T_new = relax * T_new + (1-relax) * T_old
        5. Repeat until convergence

        Parameters
        ----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance [°C]
        relax : float
            Under-relaxation factor (0 < relax <= 1)

        Returns
        -------
        T_base : array
            Base plate temperature distribution
        T_fins : list
            Temperature profiles for each fin
        Q_fins : list
            Heat removed by each fin
        """
        if verbose:
            print(f"\n{'='*60}")
            print("COUPLED HEAT SINK SOLUTION")
            print(f"{'='*60}")
            print(f"  Fins: {self.n_fins}")
            print(f"  Fin height: {self.fin_height*1000:.1f} mm")
            print(f"  Fin thickness: {self.fin_thickness*1000:.1f} mm")
            print(f"  Fin spacing: {self.fin_spacing*1000:.2f} mm")

        # Initial guess: solve base plate without fins first to get estimate
        T_base = self.plate_solver.solve_steady_state()

        # Also initialize Q_fins with estimate
        Q_fins_old = [0.0] * len(self.fin_positions)

        for iteration in range(max_iter):
            # Store old values for convergence check
            T_base_old = T_base.copy()

            # Solve each fin based on local base temperature
            self.Q_fins = []
            self.T_fins = []

            for fin_idx, fin_nodes in enumerate(self.fin_positions):
                # Get average base temperature for this fin
                node_indices = [self.grid.idx(i, j) for i, j in fin_nodes]
                T_fin_base = np.mean([T_base[n] for n in node_indices])

                # Clamp temperature to physical range
                T_fin_base = max(T_INF, min(200.0, T_fin_base))

                # Solve fin
                Q_fin, T_fin, x_fin, eta = self.solve_single_fin(T_fin_base)

                # Under-relax Q_fin
                Q_fin_relaxed = relax * Q_fin + (1 - relax) * Q_fins_old[fin_idx]

                self.Q_fins.append(Q_fin_relaxed)
                self.T_fins.append(T_fin)

            Q_fins_old = self.Q_fins.copy()

            # Update base plate solver with fin heat sinks
            self.plate_solver.fin_positions = []
            self.plate_solver.fin_heat_rates = []

            for fin_idx, fin_nodes in enumerate(self.fin_positions):
                # Distribute fin heat removal across all nodes where fin attaches
                Q_per_node = self.Q_fins[fin_idx] / len(fin_nodes)
                for i, j in fin_nodes:
                    n = self.grid.idx(i, j)
                    self.plate_solver.fin_positions.append(n)
                    self.plate_solver.fin_heat_rates.append(Q_per_node)

            # Solve base plate
            T_base_new = self.plate_solver.solve_steady_state()

            # Under-relax temperature update
            T_base = relax * T_base_new + (1 - relax) * T_base_old

            # Check convergence
            error = np.max(np.abs(T_base - T_base_old))

            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration+1}: max ΔT = {error:.4f}°C, "
                      f"T_max = {T_base.max():.2f}°C, Q_fins = {sum(self.Q_fins):.1f}W")

            if error < tol:
                self.converged = True
                if verbose:
                    print(f"  Converged after {iteration+1} iterations")
                break

        if not self.converged and verbose:
            print(f"  Warning: Did not converge after {max_iter} iterations (error={error:.4f}°C)")

        self.T_base = T_base
        return T_base, self.T_fins, self.Q_fins

    def solve_transient_coupled(self, dt=0.1, t_final=120.0, save_interval=1.0, verbose=True):
        """
        Solve coupled transient problem.

        Uses explicit coupling: at each time step, solve fins based on
        current base temperature, then advance base plate.

        Returns
        -------
        T_base_history : list
            Base plate temperature at saved times
        t_history : list
            Time values
        Q_fins_history : list
            Fin heat rates at saved times
        """
        if verbose:
            print(f"\n>>> TRANSIENT COUPLED SOLUTION <<<")
            print(f"  Simulating from t=0 to t={t_final}s...")

        # Initialize at ambient
        T_base = np.ones(self.grid.nTotal) * T_INF

        # Storage
        T_base_history = [T_base.copy()]
        Q_fins_history = [np.zeros(self.n_fins)]
        t_history = [0.0]

        # Calculate stable time step
        alpha = K / (RHO * C)
        dx_min = min(self.grid.dx, self.grid.dy)
        dt_stable = 0.25 * dx_min**2 / alpha

        if dt > dt_stable:
            print(f"  Warning: dt={dt:.6f}s > stable dt={dt_stable:.6f}s. Reducing.")
            dt = 0.9 * dt_stable

        # Precompute thermal masses
        thermal_mass = RHO * C * self.grid.volumes

        t = 0.0
        next_save = save_interval
        n_steps = int(t_final / dt)

        boundary = self.grid.get_boundary_nodes()

        for step in range(n_steps):
            # Solve fins based on current base temperature
            Q_fins_current = []
            for fin_idx, fin_nodes in enumerate(self.fin_positions):
                node_indices = [self.grid.idx(i, j) for i, j in fin_nodes]
                T_fin_base = np.mean([T_base[n] for n in node_indices])
                Q_fin, _, _, _ = self.solve_single_fin(T_fin_base)
                Q_fins_current.append(Q_fin)

            # Set fin heat rates in plate solver
            self.plate_solver.fin_positions = []
            self.plate_solver.fin_heat_rates = []

            for fin_idx, fin_nodes in enumerate(self.fin_positions):
                Q_per_node = Q_fins_current[fin_idx] / len(fin_nodes)
                for i, j in fin_nodes:
                    n = self.grid.idx(i, j)
                    self.plate_solver.fin_positions.append(n)
                    self.plate_solver.fin_heat_rates.append(Q_per_node)

            # Advance base plate one time step
            T_new = T_base.copy()

            for node in range(self.grid.nTotal):
                i, j = self.grid.ij(node)
                A_cv = self.grid.areas[node]

                # Distances
                dx_w = abs(self.grid.nodes[node, 0] - self.grid.nodes[self.grid.neighbours[node, 0], 0]) if self.grid.neighbours[node, 0] is not None else self.grid.dx
                dx_e = abs(self.grid.nodes[self.grid.neighbours[node, 1], 0] - self.grid.nodes[node, 0]) if self.grid.neighbours[node, 1] is not None else self.grid.dx
                dy_s = abs(self.grid.nodes[node, 1] - self.grid.nodes[self.grid.neighbours[node, 2], 1]) if self.grid.neighbours[node, 2] is not None else self.grid.dy
                dy_n = abs(self.grid.nodes[self.grid.neighbours[node, 3], 1] - self.grid.nodes[node, 1]) if self.grid.neighbours[node, 3] is not None else self.grid.dy

                # Interface areas
                L_w = self.grid.interface_lengths[node, 0] * self.plate_thickness
                L_e = self.grid.interface_lengths[node, 1] * self.plate_thickness
                L_s = self.grid.interface_lengths[node, 2] * self.plate_thickness
                L_n = self.grid.interface_lengths[node, 3] * self.plate_thickness

                # Conduction
                Q_cond = 0.0
                if self.grid.neighbours[node, 0] is not None and node not in boundary['west']:
                    Q_cond += K * L_w * (T_base[self.grid.neighbours[node, 0]] - T_base[node]) / dx_w
                if self.grid.neighbours[node, 1] is not None and node not in boundary['east']:
                    Q_cond += K * L_e * (T_base[self.grid.neighbours[node, 1]] - T_base[node]) / dx_e
                if self.grid.neighbours[node, 2] is not None and node not in boundary['south']:
                    Q_cond += K * L_s * (T_base[self.grid.neighbours[node, 2]] - T_base[node]) / dy_s
                if self.grid.neighbours[node, 3] is not None and node not in boundary['north']:
                    Q_cond += K * L_n * (T_base[self.grid.neighbours[node, 3]] - T_base[node]) / dy_n

                # Source from processor
                Q_source = self.plate_solver.Q_bottom * A_cv

                # Sink from fins
                if node in self.plate_solver.fin_positions:
                    idx = self.plate_solver.fin_positions.index(node)
                    Q_source -= self.plate_solver.fin_heat_rates[idx]

                # Convection from exposed top surface
                Q_conv = H * A_cv * (T_INF - T_base[node])

                # Update
                T_new[node] = T_base[node] + dt * (Q_cond + Q_source + Q_conv) / thermal_mass[node]

            T_base = T_new
            t += dt

            # Save at intervals
            if t >= next_save:
                T_base_history.append(T_base.copy())
                Q_fins_history.append(Q_fins_current.copy())
                t_history.append(t)
                next_save += save_interval

        # Save final
        if t_history[-1] < t:
            T_base_history.append(T_base.copy())
            Q_fins_history.append(Q_fins_current)
            t_history.append(t)

        self.T_base = T_base
        self.Q_fins = Q_fins_current

        if verbose:
            print(f"  Final T_max = {T_base.max():.2f}°C, Q_fins = {sum(Q_fins_current):.1f}W")

        return T_base_history, t_history, Q_fins_history

    def compute_heat_balance(self):
        """Compute and verify heat balance"""
        Q_in = Q_PROCESSOR
        Q_fins = sum(self.Q_fins) if self.Q_fins else 0.0

        # Convection from base plate (exposed areas)
        Q_conv_base = 0.0
        for n in range(self.grid.nTotal):
            # Only count areas not covered by fins
            is_fin_node = n in self.plate_solver.fin_positions
            if not is_fin_node:
                Q_conv_base += H * self.grid.areas[n] * (self.T_base[n] - T_INF)

        Q_out = Q_fins + Q_conv_base
        error = abs(Q_in - Q_out) / Q_in * 100 if Q_in > 0 else 0.0

        return Q_in, Q_fins, Q_conv_base, Q_out, error

    def compute_mass(self):
        """Calculate total heat sink mass"""
        # Base plate mass
        plate_volume = PLATE_WIDTH * PLATE_DEPTH * self.plate_thickness
        plate_mass = RHO * plate_volume

        # Fin mass (each fin)
        fin_volume = FIN_WIDTH * self.fin_height * self.fin_thickness
        fin_mass = RHO * fin_volume
        total_fin_mass = fin_mass * self.n_fins

        total_mass = plate_mass + total_fin_mass

        return {
            'plate_mass': plate_mass,
            'fin_mass': fin_mass,
            'total_fin_mass': total_fin_mass,
            'total_mass': total_mass
        }

    def print_summary(self):
        """Print summary of heat sink design"""
        mass = self.compute_mass()
        Q_in, Q_fins, Q_conv_base, Q_out, error = self.compute_heat_balance()

        print(f"\n{'='*60}")
        print("HEAT SINK DESIGN SUMMARY")
        print(f"{'='*60}")

        print(f"\nGeometry:")
        print(f"  Base plate: {PLATE_WIDTH*1000:.0f} × {PLATE_DEPTH*1000:.0f} × {self.plate_thickness*1000:.1f} mm")
        print(f"  Number of fins: {self.n_fins}")
        print(f"  Fin dimensions: {FIN_WIDTH*1000:.0f} × {self.fin_height*1000:.1f} × {self.fin_thickness*1000:.1f} mm")
        print(f"  Fin spacing: {self.fin_spacing*1000:.2f} mm")

        print(f"\nMass:")
        print(f"  Base plate: {mass['plate_mass']*1000:.1f} g")
        print(f"  Per fin: {mass['fin_mass']*1000:.2f} g")
        print(f"  Total fins: {mass['total_fin_mass']*1000:.1f} g")
        print(f"  TOTAL: {mass['total_mass']*1000:.1f} g")

        print(f"\nThermal Performance:")
        print(f"  Max base temperature: {self.T_base.max():.2f}°C")
        print(f"  Min base temperature: {self.T_base.min():.2f}°C")
        print(f"  Heat removed by fins: {Q_fins:.1f} W")
        print(f"  Heat convected (base): {Q_conv_base:.1f} W")

        print(f"\nHeat Balance:")
        print(f"  Q_in (processor): {Q_in:.1f} W")
        print(f"  Q_out (total): {Q_out:.1f} W")
        print(f"  Balance error: {error:.2f}%")

        # Check constraint
        meets_temp = self.T_base.max() <= T_MAX
        print(f"\nConstraint Check:")
        print(f"  T_max ≤ {T_MAX}°C: {'PASS' if meets_temp else 'FAIL'} ({self.T_base.max():.2f}°C)")

        print(f"{'='*60}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_heat_sink_design(hs, title="Heat Sink Design"):
    """Plot 3D-like view of heat sink with temperature"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Base plate temperature
    ax1 = axes[0]
    T_2d = hs.T_base.reshape(hs.grid.nWidth, hs.grid.nHeight)
    X = hs.grid.nodes[:, 0].reshape(hs.grid.nWidth, hs.grid.nHeight) * 1000
    Y = hs.grid.nodes[:, 1].reshape(hs.grid.nWidth, hs.grid.nHeight) * 1000

    T_range = hs.T_base.max() - hs.T_base.min()
    if T_range < 0.01:
        levels = np.linspace(hs.T_base.min() - 1, hs.T_base.max() + 1, 20)
    else:
        levels = np.linspace(hs.T_base.min(), hs.T_base.max(), 20)

    cs = ax1.contourf(X, Y, T_2d, levels=levels, cmap='hot')
    plt.colorbar(cs, ax=ax1, label='Temperature [°C]')

    # Mark fin positions
    for fin_nodes in hs.fin_positions:
        i, j = fin_nodes[0]
        x = hs.grid.nodes[hs.grid.idx(i, j), 0] * 1000
        ax1.axvline(x=x, color='blue', linestyle='--', linewidth=1, alpha=0.7)

    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title('Base Plate Temperature')
    ax1.set_aspect('equal')

    # Right: Fin temperature profiles
    ax2 = axes[1]
    if hs.T_fins:
        for i, T_fin in enumerate(hs.T_fins):
            x_fin = np.linspace(0, hs.fin_height * 1000, len(T_fin))
            ax2.plot(x_fin, T_fin, label=f'Fin {i+1}', linewidth=1.5)

    ax2.set_xlabel('Position along fin [mm]')
    ax2.set_ylabel('Temperature [°C]')
    ax2.set_title('Fin Temperature Profiles')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=T_INF, color='g', linestyle='--', label='T_ambient')
    if hs.n_fins <= 10:
        ax2.legend(loc='best', fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_transient_results(T_history, t_history, Q_history, grid, n_fins):
    """Plot transient coupled results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Temperature evolution
    ax1 = axes[0]
    # Center node
    center = grid.idx(grid.nWidth//2, grid.nHeight//2)
    T_center = [T_history[i][center] for i in range(len(T_history))]
    ax1.plot(t_history, T_center, 'b-', linewidth=2, label='Center')

    # Max temperature
    T_max = [np.max(T_history[i]) for i in range(len(T_history))]
    ax1.plot(t_history, T_max, 'r-', linewidth=2, label='Max')

    ax1.axhline(y=T_MAX, color='r', linestyle='--', alpha=0.5, label=f'Limit ({T_MAX}°C)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Temperature Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Total heat removal
    ax2 = axes[1]
    Q_total = [sum(Q_history[i]) for i in range(len(Q_history))]
    ax2.plot(t_history, Q_total, 'g-', linewidth=2)
    ax2.axhline(y=Q_PROCESSOR, color='r', linestyle='--', alpha=0.5, label=f'Q_processor ({Q_PROCESSOR}W)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Heat Removal [W]')
    ax2.set_title('Total Fin Heat Removal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("COUPLED HEAT SINK SOLVER - TP3")
    print("="*60)

    # First, verify single fin heat removal
    print("\n>>> SINGLE FIN VERIFICATION <<<")
    Q_single, T_single, x_single, eta_single = CoupledHeatSink(1).solve_single_fin(90.0)
    print(f"  Single fin at T_base=90°C removes: {Q_single:.2f} W")
    print(f"  Fin efficiency: {eta_single*100:.1f}%")
    print(f"  Fins needed (rough): {500/Q_single:.0f}")

    # Test with different number of fins
    # Need many fins since each only removes ~28W
    fin_counts = [10, 15, 18, 20, 22, 25, 30]
    results = []

    for n_fins in fin_counts:
        print(f"\n>>> Testing with {n_fins} fins <<<")

        hs = CoupledHeatSink(n_fins=n_fins)
        T_base, T_fins, Q_fins = hs.solve_steady_coupled(verbose=False, relax=0.3)

        mass = hs.compute_mass()
        T_max = T_base.max()
        Q_total = sum(Q_fins)

        results.append({
            'n_fins': n_fins,
            'T_max': T_max,
            'Q_fins': Q_total,
            'mass': mass['total_mass'] * 1000,
            'meets_constraint': T_max <= T_MAX
        })

        print(f"  T_max = {T_max:.2f}°C, Q_fins = {Q_total:.1f}W, "
              f"Mass = {mass['total_mass']*1000:.1f}g, "
              f"{'PASS' if T_max <= T_MAX else 'FAIL'}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("DESIGN COMPARISON")
    print(f"{'='*70}")
    print(f"{'N_fins':>8} {'T_max [°C]':>12} {'Q_fins [W]':>12} {'Mass [g]':>10} {'Status':>8}")
    print("-" * 55)
    for r in results:
        status = "PASS" if r['meets_constraint'] else "FAIL"
        print(f"{r['n_fins']:>8} {r['T_max']:>12.2f} {r['Q_fins']:>12.1f} {r['mass']:>10.1f} {status:>8}")

    # Find minimum mass design that meets constraint
    valid_designs = [r for r in results if r['meets_constraint']]
    if valid_designs:
        best = min(valid_designs, key=lambda x: x['mass'])
        print(f"\nBest design: {best['n_fins']} fins, mass = {best['mass']:.1f}g")

        # Full analysis of best design
        print(f"\n>>> DETAILED ANALYSIS OF BEST DESIGN <<<")
        hs_best = CoupledHeatSink(n_fins=best['n_fins'])
        hs_best.solve_steady_coupled(verbose=True)
        hs_best.print_summary()

        # Plot
        fig1 = plot_heat_sink_design(hs_best, f"Optimal Heat Sink Design ({best['n_fins']} fins)")
        fig1.savefig('heat_sink_optimal.png', dpi=150, bbox_inches='tight')
        print("\nSaved: heat_sink_optimal.png")

        # Transient analysis
        print(f"\n>>> TRANSIENT ANALYSIS <<<")
        T_history, t_history, Q_history = hs_best.solve_transient_coupled(
            dt=0.1, t_final=120.0, save_interval=1.0, verbose=True
        )

        fig2 = plot_transient_results(T_history, t_history, Q_history,
                                       hs_best.grid, best['n_fins'])
        fig2.savefig('heat_sink_transient.png', dpi=150, bbox_inches='tight')
        print("Saved: heat_sink_transient.png")

    else:
        print("\nNo design meets the temperature constraint!")
        print("Consider: more fins, taller fins, or thinner fins for better cooling")

    plt.show()
