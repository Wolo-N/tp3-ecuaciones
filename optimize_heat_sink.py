"""
Heat Sink Optimization
======================

Finds the minimum mass heat sink design that meets thermal requirements.

**Optimization approach**:
1. Pre-compute fin performance curves (Q vs T_base for different geometries)
2. Use analytical/semi-analytical model for quick evaluation
3. Search parameter space for minimum mass design

**Design Variables**:
- Number of fins (n)
- Fin thickness (t)
- Fin height (L)
- Base plate thickness (t_base)

**Constraints**:
- T_max ≤ 90°C
- Q_dissipated ≥ 500W
- Total height ≤ 25mm (fins + base)
- Fins must fit on 50mm × 50mm base
"""

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize

# =============================================================================
# PROBLEM PARAMETERS
# =============================================================================

# Target specifications
Q_TARGET = 500.0        # Heat to dissipate [W]
T_MAX = 90.0            # Maximum allowed temperature [°C]
T_INF = 45.0            # Ambient temperature [°C]
HEIGHT_MAX = 0.025      # Maximum total height [m] = 25mm

# Base plate dimensions
PLATE_WIDTH = 0.050     # [m] = 50mm
PLATE_DEPTH = 0.050     # [m] = 50mm

# Material properties - Aluminum
K = 205.0               # thermal conductivity [W/m-K]
RHO = 2700.0            # density [kg/m³]
H = 300.0               # convection coefficient [W/m²-K]

# Fin dimensions
FIN_WIDTH = PLATE_DEPTH  # Fins span the full depth


# =============================================================================
# ANALYTICAL FIN MODEL
# =============================================================================

def fin_parameter_m(h, k, thickness):
    """Calculate fin parameter m = sqrt(hP/(kA)) for rectangular fin"""
    # For rectangular fin: P ≈ 2*width (if width >> thickness)
    # A = width * thickness
    # m = sqrt(2h / (k*t))
    return sqrt(2 * h / (k * thickness))


def fin_heat_transfer(T_base, T_inf, k, h, L, width, thickness):
    """
    Calculate heat transfer from a single rectangular fin using analytical solution.

    Uses the classic fin equation solution with convective tip BC.

    Parameters
    ----------
    T_base : float
        Fin base temperature [°C]
    T_inf : float
        Ambient temperature [°C]
    k : float
        Thermal conductivity [W/m-K]
    h : float
        Convection coefficient [W/m²-K]
    L : float
        Fin length/height [m]
    width : float
        Fin width (perpendicular to heat flow) [m]
    thickness : float
        Fin thickness [m]

    Returns
    -------
    Q : float
        Heat transfer rate [W]
    eta : float
        Fin efficiency
    """
    theta_b = T_base - T_inf
    if theta_b <= 0:
        return 0.0, 0.0

    # Fin cross-section
    A_c = width * thickness

    # Perimeter (both sides of fin)
    P = 2 * (width + thickness)

    # Fin parameter
    m = sqrt(h * P / (k * A_c))

    # Corrected length for convective tip (approximation)
    L_c = L + thickness / 2

    # Heat transfer (convective tip solution simplified)
    mL = m * L_c
    if mL > 20:  # Very long fin
        Q = sqrt(h * P * k * A_c) * theta_b
    else:
        Q = sqrt(h * P * k * A_c) * theta_b * np.tanh(mL)

    # Efficiency
    Q_ideal = h * P * L * theta_b
    eta = Q / Q_ideal if Q_ideal > 0 else 0.0

    return Q, eta


def base_plate_thermal_resistance(k, thickness, width, depth, h):
    """
    Calculate base plate thermal resistance.

    Simplified model: conduction through thickness + spreading resistance
    """
    A_base = width * depth

    # Conduction resistance
    R_cond = thickness / (k * A_base)

    # Convection from bottom (processor contact - assuming perfect contact)
    # This is effectively zero for perfect contact

    return R_cond


def heat_sink_performance(n_fins, fin_height, fin_thickness, base_thickness,
                           T_base, T_inf=T_INF, k=K, h=H,
                           plate_width=PLATE_WIDTH, plate_depth=PLATE_DEPTH):
    """
    Calculate heat sink performance given design parameters.

    Returns total heat dissipated and component breakdown.
    """
    # Check geometric constraints
    if n_fins < 1:
        return 0.0, {'Q_fins': 0, 'Q_base': 0, 'valid': False}

    # Fin spacing
    fin_spacing = plate_width / n_fins

    # Check if fins fit (minimum spacing = fin thickness)
    if fin_spacing < fin_thickness:
        return 0.0, {'Q_fins': 0, 'Q_base': 0, 'valid': False}

    # Heat from each fin
    Q_per_fin, eta = fin_heat_transfer(T_base, T_inf, k, h,
                                        fin_height, plate_depth, fin_thickness)
    Q_fins = n_fins * Q_per_fin

    # Heat from exposed base plate (area not covered by fins)
    A_fins = n_fins * plate_depth * fin_thickness
    A_base_exposed = plate_width * plate_depth - A_fins
    if A_base_exposed < 0:
        A_base_exposed = 0

    Q_base = h * A_base_exposed * (T_base - T_inf)

    Q_total = Q_fins + Q_base

    info = {
        'Q_fins': Q_fins,
        'Q_base': Q_base,
        'Q_per_fin': Q_per_fin,
        'fin_efficiency': eta,
        'fin_spacing': fin_spacing,
        'valid': True
    }

    return Q_total, info


def heat_sink_mass(n_fins, fin_height, fin_thickness, base_thickness,
                   plate_width=PLATE_WIDTH, plate_depth=PLATE_DEPTH, rho=RHO):
    """Calculate total heat sink mass."""
    # Base plate mass
    V_base = plate_width * plate_depth * base_thickness
    m_base = rho * V_base

    # Fin mass
    V_fin = plate_depth * fin_height * fin_thickness
    m_fins = n_fins * rho * V_fin

    return m_base + m_fins, m_base, m_fins


def find_required_temperature(n_fins, fin_height, fin_thickness, base_thickness,
                               Q_target=Q_TARGET, T_inf=T_INF):
    """
    Find the base temperature needed to dissipate Q_target.

    Uses bisection to find T_base such that Q_total = Q_target.
    """
    T_low, T_high = T_inf + 1, 300.0

    for _ in range(50):
        T_mid = (T_low + T_high) / 2
        Q, info = heat_sink_performance(n_fins, fin_height, fin_thickness,
                                         base_thickness, T_mid, T_inf)
        if not info['valid']:
            return None

        if Q < Q_target:
            T_low = T_mid
        else:
            T_high = T_mid

        if abs(Q - Q_target) < 0.1:
            break

    return T_mid


# =============================================================================
# OPTIMIZATION
# =============================================================================

def objective_function(x, return_details=False):
    """
    Objective function for optimization.

    x = [n_fins, fin_height, fin_thickness, base_thickness]

    Returns mass if constraints satisfied, else large penalty.
    """
    n_fins = int(round(x[0]))
    fin_height = x[1]
    fin_thickness = x[2]
    base_thickness = x[3]

    # Geometric constraints
    if n_fins < 1 or fin_height <= 0 or fin_thickness <= 0 or base_thickness <= 0:
        if return_details:
            return 1e6, {'valid': False, 'reason': 'invalid dimensions'}
        return 1e6

    # Height constraint
    total_height = fin_height + base_thickness
    if total_height > HEIGHT_MAX:
        if return_details:
            return 1e6, {'valid': False, 'reason': 'exceeds height limit'}
        return 1e6

    # Spacing constraint
    fin_spacing = PLATE_WIDTH / n_fins
    if fin_spacing < fin_thickness * 1.5:  # Need some gap for air flow
        if return_details:
            return 1e6, {'valid': False, 'reason': 'fins too close'}
        return 1e6

    # Find required temperature
    T_base = find_required_temperature(n_fins, fin_height, fin_thickness, base_thickness)
    if T_base is None:
        if return_details:
            return 1e6, {'valid': False, 'reason': 'no valid temperature'}
        return 1e6

    # Temperature constraint
    if T_base > T_MAX:
        penalty = (T_base - T_MAX) * 1000
        mass, _, _ = heat_sink_mass(n_fins, fin_height, fin_thickness, base_thickness)
        if return_details:
            return mass + penalty, {'valid': False, 'reason': 'exceeds T_max',
                                    'T_base': T_base, 'mass': mass}
        return mass + penalty

    # Valid design - return mass
    mass, m_base, m_fins = heat_sink_mass(n_fins, fin_height, fin_thickness, base_thickness)

    if return_details:
        Q, info = heat_sink_performance(n_fins, fin_height, fin_thickness,
                                         base_thickness, T_base)
        return mass, {
            'valid': True,
            'n_fins': n_fins,
            'fin_height': fin_height,
            'fin_thickness': fin_thickness,
            'base_thickness': base_thickness,
            'T_base': T_base,
            'mass': mass,
            'm_base': m_base,
            'm_fins': m_fins,
            'Q_total': Q,
            'Q_fins': info['Q_fins'],
            'Q_base': info['Q_base'],
            'fin_efficiency': info['fin_efficiency'],
            'fin_spacing': info['fin_spacing']
        }
    return mass


def grid_search_optimization():
    """
    Perform grid search over design space to find optimal solution.
    """
    print("="*60)
    print("GRID SEARCH OPTIMIZATION")
    print("="*60)

    # Parameter ranges
    n_fins_range = range(5, 35, 2)
    fin_heights = np.linspace(0.010, 0.023, 7)  # Leave room for base
    fin_thicknesses = np.linspace(0.001, 0.004, 7)
    base_thicknesses = np.linspace(0.001, 0.003, 3)

    best_mass = 1e6
    best_design = None
    results = []

    total_combos = len(n_fins_range) * len(fin_heights) * len(fin_thicknesses) * len(base_thicknesses)
    print(f"Searching {total_combos} combinations...")

    count = 0
    for n_fins in n_fins_range:
        for fin_height in fin_heights:
            for fin_thickness in fin_thicknesses:
                for base_thickness in base_thicknesses:
                    count += 1

                    x = [n_fins, fin_height, fin_thickness, base_thickness]
                    mass, info = objective_function(x, return_details=True)

                    if info['valid']:
                        results.append(info)
                        if mass < best_mass:
                            best_mass = mass
                            best_design = info

    print(f"  Valid designs found: {len(results)}")

    if best_design:
        print(f"\nBest design found:")
        print(f"  Number of fins: {best_design['n_fins']}")
        print(f"  Fin height: {best_design['fin_height']*1000:.2f} mm")
        print(f"  Fin thickness: {best_design['fin_thickness']*1000:.2f} mm")
        print(f"  Base thickness: {best_design['base_thickness']*1000:.2f} mm")
        print(f"  Fin spacing: {best_design['fin_spacing']*1000:.2f} mm")
        print(f"  Total height: {(best_design['fin_height'] + best_design['base_thickness'])*1000:.2f} mm")
        print(f"  Base temperature: {best_design['T_base']:.2f}°C")
        print(f"  Total mass: {best_design['mass']*1000:.1f} g")
        print(f"  Fin efficiency: {best_design['fin_efficiency']*100:.1f}%")
    else:
        print("\nNo valid design found!")
        print("Consider relaxing constraints or increasing cooling capacity.")

    return best_design, results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_design_space(results, best_design):
    """Plot the design space exploration results."""
    if not results:
        print("No valid results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Filter valid results
    valid = [r for r in results if r['valid']]

    if not valid:
        print("No valid designs to plot")
        return

    # Extract data
    n_fins = [r['n_fins'] for r in valid]
    masses = [r['mass']*1000 for r in valid]
    T_bases = [r['T_base'] for r in valid]
    heights = [r['fin_height']*1000 for r in valid]
    thicknesses = [r['fin_thickness']*1000 for r in valid]

    # Plot 1: Mass vs N_fins
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(n_fins, masses, c=T_bases, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter1, ax=ax1, label='T_base [°C]')
    if best_design:
        ax1.scatter(best_design['n_fins'], best_design['mass']*1000,
                   marker='*', s=200, c='green', edgecolor='black', label='Optimal')
    ax1.set_xlabel('Number of fins')
    ax1.set_ylabel('Mass [g]')
    ax1.set_title('Mass vs Number of Fins')
    ax1.axhline(y=T_MAX, color='r', linestyle='--', alpha=0.3)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mass vs Fin height
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(heights, masses, c=n_fins, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter2, ax=ax2, label='N_fins')
    if best_design:
        ax2.scatter(best_design['fin_height']*1000, best_design['mass']*1000,
                   marker='*', s=200, c='green', edgecolor='black')
    ax2.set_xlabel('Fin height [mm]')
    ax2.set_ylabel('Mass [g]')
    ax2.set_title('Mass vs Fin Height')
    ax2.grid(True, alpha=0.3)

    # Plot 3: T_base vs N_fins
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(n_fins, T_bases, c=masses, cmap='plasma', alpha=0.7)
    plt.colorbar(scatter3, ax=ax3, label='Mass [g]')
    ax3.axhline(y=T_MAX, color='r', linestyle='--', label=f'T_max = {T_MAX}°C')
    if best_design:
        ax3.scatter(best_design['n_fins'], best_design['T_base'],
                   marker='*', s=200, c='green', edgecolor='black')
    ax3.set_xlabel('Number of fins')
    ax3.set_ylabel('Base temperature [°C]')
    ax3.set_title('Temperature vs Number of Fins')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Pareto front (Mass vs T_base)
    ax4 = axes[1, 1]
    ax4.scatter(T_bases, masses, c=n_fins, cmap='viridis', alpha=0.7)
    ax4.axvline(x=T_MAX, color='r', linestyle='--', label=f'T_max = {T_MAX}°C')
    if best_design:
        ax4.scatter(best_design['T_base'], best_design['mass']*1000,
                   marker='*', s=200, c='green', edgecolor='black', label='Optimal')
    ax4.set_xlabel('Base temperature [°C]')
    ax4.set_ylabel('Mass [g]')
    ax4.set_title('Pareto Front')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_optimal_design(design):
    """Plot schematic of optimal design."""
    if not design:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw base plate (cross-section view)
    base_w = design['fin_spacing'] * design['n_fins'] * 1000
    base_h = design['base_thickness'] * 1000

    base = plt.Rectangle((0, 0), base_w, base_h, facecolor='lightblue',
                          edgecolor='blue', linewidth=2)
    ax.add_patch(base)

    # Draw fins
    fin_h = design['fin_height'] * 1000
    fin_t = design['fin_thickness'] * 1000
    spacing = design['fin_spacing'] * 1000

    for i in range(design['n_fins']):
        x = i * spacing + (spacing - fin_t) / 2
        fin = plt.Rectangle((x, base_h), fin_t, fin_h, facecolor='lightcoral',
                             edgecolor='red', linewidth=1)
        ax.add_patch(fin)

    # Labels
    ax.set_xlim(-2, base_w + 2)
    ax.set_ylim(-2, base_h + fin_h + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('Width [mm]')
    ax.set_ylabel('Height [mm]')
    ax.set_title(f'Optimal Heat Sink Design\n'
                f'{design["n_fins"]} fins, {design["mass"]*1000:.1f}g, T_base={design["T_base"]:.1f}°C')
    ax.grid(True, alpha=0.3)

    # Dimensions
    ax.annotate(f'{base_w:.1f} mm', xy=(base_w/2, -1), ha='center')
    ax.annotate(f'{base_h:.1f} mm', xy=(-1, base_h/2), ha='right', rotation=90, va='center')
    ax.annotate(f'{fin_h:.1f} mm', xy=(base_w + 1, base_h + fin_h/2), ha='left', rotation=90, va='center')

    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("HEAT SINK OPTIMIZATION - TP3")
    print("="*60)
    print(f"\nTarget: Dissipate {Q_TARGET}W with T_max ≤ {T_MAX}°C")
    print(f"Constraints: Total height ≤ {HEIGHT_MAX*1000}mm, Base {PLATE_WIDTH*1000}×{PLATE_DEPTH*1000}mm")

    # Quick analytical check
    print("\n>>> ANALYTICAL ESTIMATES <<<")
    # Single fin at T_max
    Q_single, eta = fin_heat_transfer(T_MAX, T_INF, K, H, 0.020, FIN_WIDTH, 0.002)
    print(f"Single 20mm×2mm fin at {T_MAX}°C: Q = {Q_single:.2f}W, η = {eta*100:.1f}%")
    print(f"Estimated fins needed: {Q_TARGET/Q_single:.0f}")

    # Run optimization
    best_design, results = grid_search_optimization()

    if best_design:
        # Plot results
        fig1 = plot_design_space(results, best_design)
        fig1.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
        print("\nSaved: optimization_results.png")

        fig2 = plot_optimal_design(best_design)
        fig2.savefig('optimal_design.png', dpi=150, bbox_inches='tight')
        print("Saved: optimal_design.png")

        # Print final summary
        print("\n" + "="*60)
        print("OPTIMAL DESIGN SUMMARY")
        print("="*60)
        print(f"\nGeometry:")
        print(f"  Number of fins: {best_design['n_fins']}")
        print(f"  Fin height: {best_design['fin_height']*1000:.2f} mm")
        print(f"  Fin thickness: {best_design['fin_thickness']*1000:.2f} mm")
        print(f"  Fin spacing: {best_design['fin_spacing']*1000:.2f} mm")
        print(f"  Base thickness: {best_design['base_thickness']*1000:.2f} mm")
        print(f"  Total height: {(best_design['fin_height']+best_design['base_thickness'])*1000:.2f} mm")

        print(f"\nMass:")
        print(f"  Base plate: {best_design['m_base']*1000:.2f} g")
        print(f"  Fins (total): {best_design['m_fins']*1000:.2f} g")
        print(f"  TOTAL: {best_design['mass']*1000:.2f} g")

        print(f"\nThermal Performance:")
        print(f"  Base temperature: {best_design['T_base']:.2f}°C (limit: {T_MAX}°C)")
        print(f"  Heat dissipated: {best_design['Q_total']:.1f} W (target: {Q_TARGET}W)")
        print(f"  Heat from fins: {best_design['Q_fins']:.1f} W ({best_design['Q_fins']/best_design['Q_total']*100:.1f}%)")
        print(f"  Heat from base: {best_design['Q_base']:.1f} W ({best_design['Q_base']/best_design['Q_total']*100:.1f}%)")
        print(f"  Fin efficiency: {best_design['fin_efficiency']*100:.1f}%")

        print("="*60)

    plt.show()
