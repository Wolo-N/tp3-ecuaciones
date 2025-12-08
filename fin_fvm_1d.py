"""
1D Finite Volume Method Solver for Heat Sink Fins
==================================================

**Governing equation (transient)**:
$$ k\,A_W\,\frac{T_W - T_b}{dx} + h_b \Delta x \,P_b\,\big(T_{\infty} - T_b \big) + k\, A_E\,\frac{T_E -T_b }{\Delta x}= \rho A(x) C \frac{T_b^{Next} - T_b}{\Delta t}$$

**Features**
- Variable **geometry** via user-defined `A(x)` and `P(x)` (rectangular or tapered fins)
- Supports both **steady-state** and **transient** (Forward Euler) solutions
- Base boundary at `x=0`: **Dirichlet** temperature
- Tip at `x=L`: **convective (Robin)** boundary condition
- Computes heat balance verification and fin efficiency
- Designed for heat sink optimization (TP3 project)

**Problem specs (TP3)**:
- Dissipate 500W from 50mm × 50mm processor
- Max temperature: 90°C
- Material: Aluminum
- h = 300 W/m²K
- Max height: 25mm
"""

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETERS - Updated for TP3 Heat Sink Design
# =============================================================================

# Fin geometry
lFin = 0.025          # fin height [m] (max 25mm per specs)
nVolumes = 20         # number of control volumes (nodes)

# Boundary Conditions
T_inf = 45.0          # ambient temperature [°C] (worst case operating temp)
T_base = 90.0         # base temperature [°C] (max allowed processor temp)

# Material properties - Aluminum
k = 205.0             # thermal conductivity [W/m-K]
h = 300.0             # convection coefficient [W/m²-K] (per specs)
c = 902.0             # specific heat [J/kg/K]
rho = 2700.0          # density [kg/m³]

# Fin geometry type: 'rectangular' or 'tapered'
FIN_TYPE = 'rectangular'

# Rectangular fin dimensions (flat fin for heat sink)
fin_width = 0.050     # fin width (perpendicular to flow) [m] - 50mm base
fin_thickness = 0.002 # fin thickness [m] - 2mm

# Tapered elliptic fin (alternative geometry)
R0 = 0.0025    # base major radius [m]
r0 = 0.001     # base minor radius [m]
RL = 0.001     # tip major radius [m]
rL = 0.001     # tip minor radius [m]

# Transient simulation parameters
dt = 0.01             # time step [s]
t_final = 60.0        # simulation time [s]
save_interval = 1.0   # save temperature every N seconds


def r_fun(x, lFin_val=None):
    """Tapered elliptic fin radius function"""
    L = lFin_val if lFin_val is not None else lFin
    return (r0 + (rL - r0) * (x / L), R0 + (RL - R0) * (x / L))


def A_fun(x, fin_type=None, width=None, thickness=None):
    """Cross-sectional area as function of position"""
    ft = fin_type if fin_type is not None else FIN_TYPE
    w = width if width is not None else fin_width
    t = thickness if thickness is not None else fin_thickness

    if ft == 'rectangular':
        return w * t
    else:  # tapered
        (R, r) = r_fun(x)
        return pi * R * r


def P_fun(x, fin_type=None, width=None, thickness=None):
    """Perimeter as function of position"""
    ft = fin_type if fin_type is not None else FIN_TYPE
    w = width if width is not None else fin_width
    t = thickness if thickness is not None else fin_thickness

    if ft == 'rectangular':
        return 2 * (w + t)
    else:  # tapered
        (R, r) = r_fun(x)
        return 2.0 * sqrt((R**2 + r**2) / 2)

# =============================================================================
# FINITE VOLUME SOLVER
# =============================================================================

def build_grid(lFin, nVolumes):
    """
    Build 1D grid for fin analysis.

    Returns:
        x: node positions [m]
        areas: [A_W, A_E, P*dx] for each volume
        neighbours: [W, P, E] indices for each node
        dx: grid spacing [m]
        volumes: control volume sizes [m³]
    """
    x = np.linspace(0.0, lFin, nVolumes)
    dx = x[1] - x[0]

    # Neighbours: [West, Self, East]
    neighbours = np.empty((nVolumes, 3), dtype=object)
    neighbours[0, :] = [None, 0, 1]                               # first volume
    neighbours[nVolumes-1, :] = [nVolumes-2, nVolumes-1, None]    # last volume
    for i in range(1, nVolumes-1):                                # internal volumes
        neighbours[i, :] = [i-1, i, i+1]

    # Areas: [A_W, A_E, P*dx (lateral surface area)]
    areas = np.empty((nVolumes+1, 3))
    areas[0, :] = [A_fun(0), A_fun(dx), P_fun(dx/2)*dx]
    areas[nVolumes, :] = [A_fun(lFin-dx), A_fun(lFin), P_fun(lFin-dx/2)*dx]
    for i in range(1, nVolumes):
        areas[i, :] = [A_fun(i*dx), A_fun((i+1)*dx), P_fun((i+0.5)*dx)*dx]

    # Control volumes for transient analysis
    volumes = np.zeros(nVolumes)
    for i in range(nVolumes):
        A_avg = (A_fun(i*dx) + A_fun((i+1)*dx)) / 2 if i < nVolumes-1 else A_fun(lFin)
        volumes[i] = A_avg * dx

    return x, areas, neighbours, dx, volumes


def assemble_steady_system(areas, neighbours, nVolumes, dx, T_base_val, T_inf_val, k_val, h_val):
    """
    Assemble system matrix for steady-state solution.

    Returns:
        M: coefficient matrix
        V: RHS vector
    """
    M = np.zeros((nVolumes, nVolumes), dtype=float)
    V = np.zeros(nVolumes, dtype=float)

    # Interior nodes
    for i in range(1, nVolumes-1):
        M[i, neighbours[i, 0]] = k_val / dx * areas[i, 0]
        M[i, neighbours[i, 2]] = k_val / dx * areas[i, 1]
        M[i, neighbours[i, 1]] = k_val / dx * (-areas[i, 0] - areas[i, 1]) - h_val * areas[i, 2]
        V[i] = -h_val * areas[i, 2] * T_inf_val

    # Left boundary: fixed Temperature (Dirichlet)
    M[0, 0] = 1.0
    V[0] = T_base_val

    # Right boundary: convective (Robin) at tip
    M[nVolumes-1, neighbours[nVolumes-1, 0]] = k_val / dx * areas[nVolumes-1, 0]
    M[nVolumes-1, neighbours[nVolumes-1, 1]] = (k_val / dx * (-areas[nVolumes-1, 0])
                                                 - h_val * (areas[nVolumes-1, 1] + areas[nVolumes-1, 2] / 2))
    V[nVolumes-1] = -h_val * (areas[nVolumes-1, 1] + areas[nVolumes-1, 2] / 2) * T_inf_val

    return M, V


def solve_steady_state(lFin_val=None, nVolumes_val=None, T_base_val=None, T_inf_val=None,
                       k_val=None, h_val=None):
    """
    Solve steady-state temperature distribution.

    Returns:
        T: temperature array [°C]
        x: position array [m]
        Q_base: heat entering at base [W]
        Q_conv_total: total convected heat [W]
        fin_efficiency: fin efficiency [-]
    """
    # Use defaults if not specified
    lFin_val = lFin_val if lFin_val is not None else lFin
    nVolumes_val = nVolumes_val if nVolumes_val is not None else nVolumes
    T_base_val = T_base_val if T_base_val is not None else T_base
    T_inf_val = T_inf_val if T_inf_val is not None else T_inf
    k_val = k_val if k_val is not None else k
    h_val = h_val if h_val is not None else h

    # Build grid
    x, areas, neighbours, dx, volumes = build_grid(lFin_val, nVolumes_val)

    # Assemble and solve
    M, V = assemble_steady_system(areas, neighbours, nVolumes_val, dx, T_base_val, T_inf_val, k_val, h_val)
    T = np.linalg.solve(M, V)

    # Calculate heat balance
    Q_base, Q_conv_total, fin_efficiency = calculate_heat_balance(
        T, x, areas, dx, T_base_val, T_inf_val, k_val, h_val, lFin_val
    )

    return T, x, Q_base, Q_conv_total, fin_efficiency


def solve_transient(lFin_val=None, nVolumes_val=None, T_base_val=None, T_inf_val=None,
                    k_val=None, h_val=None, c_val=None, rho_val=None,
                    dt_val=None, t_final_val=None, T_init=None):
    """
    Solve transient temperature evolution using Forward Euler.

    Returns:
        T_history: list of temperature arrays at saved times
        t_history: list of time values [s]
        x: position array [m]
        T_final: final temperature distribution
    """
    # Use defaults if not specified
    lFin_val = lFin_val if lFin_val is not None else lFin
    nVolumes_val = nVolumes_val if nVolumes_val is not None else nVolumes
    T_base_val = T_base_val if T_base_val is not None else T_base
    T_inf_val = T_inf_val if T_inf_val is not None else T_inf
    k_val = k_val if k_val is not None else k
    h_val = h_val if h_val is not None else h
    c_val = c_val if c_val is not None else c
    rho_val = rho_val if rho_val is not None else rho
    dt_val = dt_val if dt_val is not None else dt
    t_final_val = t_final_val if t_final_val is not None else t_final

    # Build grid
    x, areas, neighbours, dx, volumes = build_grid(lFin_val, nVolumes_val)

    # Initial condition
    if T_init is None:
        T = np.ones(nVolumes_val) * T_inf_val  # Start at ambient
    else:
        T = T_init.copy()
    T[0] = T_base_val  # Base always at T_base

    # Calculate stable time step (CFL condition)
    alpha = k_val / (rho_val * c_val)  # thermal diffusivity
    dt_stable = 0.25 * dx**2 / alpha  # stability criterion
    if dt_val > dt_stable:
        print(f"Warning: dt={dt_val:.4f}s > stable dt={dt_stable:.4f}s. Reducing time step.")
        dt_val = 0.9 * dt_stable

    # Time stepping
    T_history = [T.copy()]
    t_history = [0.0]
    t = 0.0
    next_save = save_interval

    n_steps = int(t_final_val / dt_val)

    for step in range(n_steps):
        T_new = T.copy()

        # Interior nodes: Forward Euler
        for i in range(1, nVolumes_val - 1):
            # Conduction terms
            Q_W = k_val * areas[i, 0] * (T[i-1] - T[i]) / dx
            Q_E = k_val * areas[i, 1] * (T[i+1] - T[i]) / dx
            # Convection term
            Q_conv = h_val * areas[i, 2] * (T_inf_val - T[i])
            # Thermal mass
            thermal_mass = rho_val * volumes[i] * c_val
            # Update temperature
            T_new[i] = T[i] + dt_val * (Q_W + Q_E + Q_conv) / thermal_mass

        # Tip node (convective BC)
        i = nVolumes_val - 1
        Q_W = k_val * areas[i, 0] * (T[i-1] - T[i]) / dx
        Q_conv_lat = h_val * areas[i, 2] / 2 * (T_inf_val - T[i])  # half lateral
        Q_conv_tip = h_val * areas[i, 1] * (T_inf_val - T[i])      # tip
        thermal_mass = rho_val * volumes[i] * c_val
        T_new[i] = T[i] + dt_val * (Q_W + Q_conv_lat + Q_conv_tip) / thermal_mass

        # Base BC (Dirichlet)
        T_new[0] = T_base_val

        T = T_new
        t += dt_val

        # Save at intervals
        if t >= next_save:
            T_history.append(T.copy())
            t_history.append(t)
            next_save += save_interval

    # Ensure final state is saved
    if t_history[-1] < t:
        T_history.append(T.copy())
        t_history.append(t)

    return T_history, t_history, x, T


def calculate_heat_balance(T, x, areas, dx, T_base_val, T_inf_val, k_val, h_val, lFin_val):
    """
    Calculate heat balance verification and fin efficiency.

    Returns:
        Q_base: heat entering at fin base [W]
        Q_conv_total: total heat convected from fin [W]
        fin_efficiency: fin efficiency [-]
    """
    nVolumes = len(T)

    # Heat entering at base (conduction from base into fin)
    Q_base = k_val * areas[0, 1] * (T[0] - T[1]) / dx

    # Total convected heat
    Q_conv_total = 0.0

    # Lateral surfaces
    for i in range(nVolumes):
        if i == 0:
            Q_conv_total += h_val * (areas[i, 2] / 2) * (T[i] - T_inf_val)
        elif i == nVolumes - 1:
            Q_conv_total += h_val * (areas[i, 2] / 2) * (T[i] - T_inf_val)
            # Tip surface
            Q_conv_total += h_val * areas[i, 1] * (T[i] - T_inf_val)
        else:
            Q_conv_total += h_val * areas[i, 2] * (T[i] - T_inf_val)

    # Fin efficiency: actual heat / ideal heat (if entire fin at T_base)
    # Ideal: all surface at T_base
    total_surface = sum(areas[i, 2] for i in range(nVolumes)) + areas[nVolumes-1, 1]
    Q_ideal = h_val * total_surface * (T_base_val - T_inf_val)
    fin_efficiency = Q_conv_total / Q_ideal if Q_ideal > 0 else 0.0

    return Q_base, Q_conv_total, fin_efficiency


def calculate_fin_mass(lFin_val=None, rho_val=None):
    """Calculate fin mass [kg]"""
    lFin_val = lFin_val if lFin_val is not None else lFin
    rho_val = rho_val if rho_val is not None else rho

    # Integrate cross-sectional area along length
    n_points = 100
    x_pts = np.linspace(0, lFin_val, n_points)
    dx_int = x_pts[1] - x_pts[0]

    volume = sum(A_fun(x_pt) * dx_int for x_pt in x_pts)
    mass = rho_val * volume

    return mass, volume


def print_results(T, x, Q_base, Q_conv_total, fin_efficiency, lFin_val=None):
    """Print summary of results"""
    lFin_val = lFin_val if lFin_val is not None else lFin

    mass, volume = calculate_fin_mass(lFin_val)

    print("\n" + "="*60)
    print("1D FIN ANALYSIS RESULTS")
    print("="*60)
    print(f"\nGeometry:")
    print(f"  Fin type: {FIN_TYPE}")
    print(f"  Fin height: {lFin_val*1000:.1f} mm")
    if FIN_TYPE == 'rectangular':
        print(f"  Fin width: {fin_width*1000:.1f} mm")
        print(f"  Fin thickness: {fin_thickness*1000:.1f} mm")
    print(f"  Fin volume: {volume*1e6:.2f} mm³")
    print(f"  Fin mass: {mass*1000:.3f} g")

    print(f"\nTemperatures:")
    print(f"  Base (x=0): {T[0]:.2f} °C")
    print(f"  Tip (x=L): {T[-1]:.2f} °C")
    print(f"  Temperature drop: {T[0] - T[-1]:.2f} °C")

    print(f"\nHeat Balance:")
    print(f"  Heat entering at base: Q_base = {Q_base:.4f} W")
    print(f"  Heat convected (total): Q_conv = {Q_conv_total:.4f} W")
    print(f"  Balance error: {abs(Q_base - Q_conv_total)/Q_base*100:.2f}%")

    print(f"\nPerformance:")
    print(f"  Fin efficiency: η = {fin_efficiency*100:.1f}%")
    print(f"  Heat per unit mass: {Q_conv_total/(mass*1000):.2f} W/g")
    print("="*60)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_steady_state(T, x, title="Steady-State Temperature Distribution"):
    """Plot steady-state temperature profile"""
    plt.figure(figsize=(10, 5))
    plt.plot(x * 1000, T, 'b-o', linewidth=2, markersize=4)
    plt.xlabel("Position along fin [mm]")
    plt.ylabel("Temperature [°C]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=T_inf, color='g', linestyle='--', label=f'T_ambient = {T_inf}°C')
    plt.axhline(y=T_base, color='r', linestyle='--', label=f'T_base = {T_base}°C')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_transient(T_history, t_history, x, title="Transient Temperature Evolution"):
    """Plot transient temperature evolution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Temperature profiles at different times
    n_profiles = min(10, len(T_history))
    indices = np.linspace(0, len(T_history)-1, n_profiles, dtype=int)
    colors = plt.cm.hot(np.linspace(0.2, 0.8, n_profiles))

    for i, idx in enumerate(indices):
        ax1.plot(x * 1000, T_history[idx], color=colors[i],
                 label=f't = {t_history[idx]:.1f}s', linewidth=1.5)

    ax1.set_xlabel("Position along fin [mm]")
    ax1.set_ylabel("Temperature [°C]")
    ax1.set_title("Temperature Profiles Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8)
    ax1.axhline(y=T_inf, color='g', linestyle='--', alpha=0.5)

    # Right: Temperature at specific locations vs time
    locations = [0, len(x)//4, len(x)//2, 3*len(x)//4, -1]
    for loc in locations:
        T_at_loc = [T_history[i][loc] for i in range(len(T_history))]
        ax2.plot(t_history, T_at_loc, linewidth=1.5,
                 label=f'x = {x[loc]*1000:.1f} mm')

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Temperature [°C]")
    ax2.set_title("Temperature vs Time at Different Locations")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("1D FIN ANALYSIS - HEAT SINK DESIGN (TP3)")
    print("="*60)

    # --- STEADY STATE ANALYSIS ---
    print("\n>>> STEADY-STATE ANALYSIS <<<")
    T, x, Q_base, Q_conv_total, fin_efficiency = solve_steady_state()
    print_results(T, x, Q_base, Q_conv_total, fin_efficiency)

    # Plot steady state
    fig1 = plot_steady_state(T, x)
    fig1.savefig('fin_steady_state.png', dpi=150, bbox_inches='tight')
    print("\nSteady-state plot saved as 'fin_steady_state.png'")

    # --- TRANSIENT ANALYSIS ---
    print("\n>>> TRANSIENT ANALYSIS <<<")
    print(f"Simulating from t=0 to t={t_final}s...")

    T_history, t_history, x_trans, T_final = solve_transient()

    print(f"\nTransient simulation complete:")
    print(f"  Time steps saved: {len(t_history)}")
    print(f"  Final time: {t_history[-1]:.1f}s")
    print(f"  Final tip temperature: {T_final[-1]:.2f}°C")

    # Check convergence to steady state
    T_ss = T  # Steady state solution
    error = np.max(np.abs(T_final - T_ss))
    print(f"  Max difference from steady state: {error:.4f}°C")

    # Plot transient
    fig2 = plot_transient(T_history, t_history, x_trans)
    fig2.savefig('fin_transient.png', dpi=150, bbox_inches='tight')
    print("\nTransient plot saved as 'fin_transient.png'")

    # --- PARAMETRIC STUDY ---
    print("\n>>> PARAMETRIC STUDY: Heat per fin vs. fin height <<<")
    heights = np.linspace(0.005, 0.025, 5)  # 5mm to 25mm
    Q_per_fin = []
    efficiencies = []
    masses = []

    for L in heights:
        T_temp, _, Q_b, Q_c, eff = solve_steady_state(lFin_val=L)
        mass, _ = calculate_fin_mass(lFin_val=L)
        Q_per_fin.append(Q_c)
        efficiencies.append(eff)
        masses.append(mass)

    print(f"\n{'Height [mm]':>12} {'Q [W]':>10} {'η [%]':>8} {'Mass [g]':>10} {'Q/mass [W/g]':>12}")
    print("-" * 55)
    for i, L in enumerate(heights):
        print(f"{L*1000:>12.1f} {Q_per_fin[i]:>10.3f} {efficiencies[i]*100:>8.1f} {masses[i]*1000:>10.3f} {Q_per_fin[i]/(masses[i]*1000):>12.2f}")

    # Estimate number of fins needed for 500W
    print(f"\n>>> FINS REQUIRED FOR 500W <<<")
    Q_target = 500.0
    for i, L in enumerate(heights):
        n_fins = np.ceil(Q_target / Q_per_fin[i])
        total_mass = n_fins * masses[i] * 1000  # grams
        print(f"  Height {L*1000:.0f}mm: {int(n_fins)} fins, total mass = {total_mass:.1f}g")

    plt.show()