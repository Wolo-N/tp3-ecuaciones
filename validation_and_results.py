"""
Validation and Final Results - TP3 Heat Sink Design
====================================================

This script generates all deliverables required for the TP3 project:

**Deliverables**:
1. Fin geometry
2. Number of fins
3. Heat sink mass
4. Thermal evolution from startup to steady state

**Verification**:
- Heat balance (Q_in = Q_out)
- FVM code validation
"""

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Import our modules
from fin_fvm_1d import (solve_steady_state as solve_fin_steady,
                        solve_transient as solve_fin_transient,
                        calculate_fin_mass, A_fun, P_fun)
from optimize_heat_sink import (fin_heat_transfer, heat_sink_performance,
                                heat_sink_mass, find_required_temperature)

# =============================================================================
# OPTIMAL DESIGN PARAMETERS (from optimization)
# =============================================================================

# Material - Aluminum
K = 205.0        # thermal conductivity [W/m-K]
RHO = 2700.0     # density [kg/m³]
C = 902.0        # specific heat [J/kg-K]
H = 300.0        # convection coefficient [W/m²-K]

# Operating conditions
Q_TARGET = 500.0  # Heat to dissipate [W]
T_INF = 45.0      # Ambient temperature [°C]
T_MAX = 90.0      # Maximum allowed temperature [°C]

# Optimal design from optimization
OPTIMAL_DESIGN = {
    'n_fins': 33,
    'fin_height': 0.01217,      # 12.17 mm
    'fin_thickness': 0.001,     # 1 mm
    'fin_width': 0.050,         # 50 mm (full depth)
    'base_thickness': 0.001,    # 1 mm
    'plate_width': 0.050,       # 50 mm
    'plate_depth': 0.050,       # 50 mm
}


# =============================================================================
# CODE VALIDATION
# =============================================================================

def validate_fvm_code():
    """
    Validate FVM code against analytical solutions.
    """
    print("="*60)
    print("CODE VALIDATION")
    print("="*60)

    # Test case: fin with known analytical solution
    # Short fin with insulated tip: T(x) = T_inf + (T_b - T_inf) * cosh(m(L-x))/cosh(mL)
    # Q = sqrt(hPkA) * (T_b - T_inf) * tanh(mL)

    L = 0.02       # 20mm
    t = 0.002      # 2mm
    w = 0.05       # 50mm
    T_base = 90.0
    T_inf_test = 45.0

    # Analytical solution
    A_c = w * t
    P = 2 * (w + t)
    m = sqrt(H * P / (K * A_c))
    mL = m * L

    theta_b = T_base - T_inf_test
    Q_analytical = sqrt(H * P * K * A_c) * theta_b * np.tanh(mL)

    # Numerical solution (FVM)
    T_num, x_num, Q_base_num, Q_conv_num, eta_num = solve_fin_steady(
        lFin_val=L,
        nVolumes_val=50,
        T_base_val=T_base,
        T_inf_val=T_inf_test,
        k_val=K,
        h_val=H
    )

    print(f"\nTest: 20mm × 2mm × 50mm rectangular fin")
    print(f"  T_base = {T_base}°C, T_inf = {T_inf_test}°C")
    print(f"\n  Analytical Q = {Q_analytical:.4f} W")
    print(f"  FVM Q_conv  = {Q_conv_num:.4f} W")
    print(f"  Error: {abs(Q_analytical - Q_conv_num)/Q_analytical*100:.2f}%")

    # Temperature profile comparison
    T_analytical = T_inf_test + theta_b * np.cosh(m * (L - x_num)) / np.cosh(mL)

    print(f"\n  Temperature at tip:")
    print(f"    Analytical: {T_analytical[-1]:.2f}°C")
    print(f"    FVM:        {T_num[-1]:.2f}°C")
    print(f"    Error:      {abs(T_analytical[-1] - T_num[-1]):.2f}°C")

    return Q_analytical, Q_conv_num, T_analytical, T_num, x_num


def verify_heat_balance(design):
    """
    Verify heat balance: Q_in = Q_out
    """
    print("\n" + "="*60)
    print("HEAT BALANCE VERIFICATION")
    print("="*60)

    # Calculate heat dissipated
    T_base = find_required_temperature(
        design['n_fins'],
        design['fin_height'],
        design['fin_thickness'],
        design['base_thickness']
    )

    Q_total, info = heat_sink_performance(
        design['n_fins'],
        design['fin_height'],
        design['fin_thickness'],
        design['base_thickness'],
        T_base
    )

    print(f"\nOperating point: T_base = {T_base:.2f}°C")
    print(f"\nHeat IN:")
    print(f"  Processor heat: {Q_TARGET:.1f} W")

    print(f"\nHeat OUT:")
    print(f"  Fins (convection): {info['Q_fins']:.1f} W")
    print(f"  Base (convection): {info['Q_base']:.1f} W")
    print(f"  Total OUT: {Q_total:.1f} W")

    error = abs(Q_TARGET - Q_total) / Q_TARGET * 100
    print(f"\nBalance error: {error:.2f}%")

    return T_base, Q_total, info


# =============================================================================
# TRANSIENT ANALYSIS
# =============================================================================

def simulate_transient_startup(design, t_final=300.0):
    """
    Simulate heat sink startup from ambient to steady state.
    """
    print("\n" + "="*60)
    print("TRANSIENT SIMULATION")
    print("="*60)
    print(f"Simulating from t=0 to t={t_final}s...")

    # Simulate single representative fin
    T_history, t_history, x_fin, T_final = solve_fin_transient(
        lFin_val=design['fin_height'],
        nVolumes_val=30,
        T_base_val=T_MAX,  # Assuming base reaches T_max at steady state
        T_inf_val=T_INF,
        k_val=K,
        h_val=H,
        c_val=C,
        rho_val=RHO,
        dt_val=0.01,
        t_final_val=t_final
    )

    # Calculate thermal time constant
    # Approximate: tau = (rho * V * C) / (h * A_s)
    V_fin = design['fin_width'] * design['fin_height'] * design['fin_thickness']
    A_s = 2 * design['fin_width'] * design['fin_height']  # Both sides
    tau = (RHO * V_fin * C) / (H * A_s)

    print(f"\n  Thermal time constant (single fin): τ ≈ {tau:.1f} s")
    print(f"  99% steady state at: ~5τ = {5*tau:.0f} s")

    # Check convergence
    T_ss, _, _, _, _ = solve_fin_steady(
        lFin_val=design['fin_height'],
        nVolumes_val=30,  # Match transient grid
        T_base_val=T_MAX,
        T_inf_val=T_INF,
        k_val=K,
        h_val=H
    )

    error = np.max(np.abs(T_final - T_ss))
    print(f"  Max error from steady state at t={t_final}s: {error:.4f}°C")

    return T_history, t_history, x_fin, tau


# =============================================================================
# GENERATE FIGURES
# =============================================================================

def plot_validation_comparison(T_analytical, T_num, x):
    """Plot FVM vs analytical comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x * 1000, T_analytical, 'b-', linewidth=2, label='Analytical')
    ax.plot(x * 1000, T_num, 'ro', markersize=6, label='FVM', markerfacecolor='none')

    ax.set_xlabel('Position along fin [mm]')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title('Code Validation: FVM vs Analytical Solution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_transient_evolution(T_history, t_history, x_fin, tau):
    """Plot transient temperature evolution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Temperature profiles at different times
    ax1 = axes[0]
    n_profiles = min(8, len(T_history))
    indices = [0] + list(np.linspace(1, len(T_history)-1, n_profiles-1, dtype=int))
    colors = plt.cm.hot(np.linspace(0.2, 0.8, len(indices)))

    for i, idx in enumerate(indices):
        ax1.plot(x_fin * 1000, T_history[idx], color=colors[i],
                 label=f't = {t_history[idx]:.0f}s', linewidth=1.5)

    ax1.set_xlabel('Position along fin [mm]')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Temperature Profiles During Startup')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=T_INF, color='g', linestyle='--', alpha=0.5, label='T_ambient')

    # Right: Temperature at specific locations vs time
    ax2 = axes[1]
    locations = {'Base (x=0)': 0, 'Middle': len(x_fin)//2, 'Tip': -1}

    for label, loc in locations.items():
        T_at_loc = [T_history[i][loc] for i in range(len(T_history))]
        ax2.plot(t_history, T_at_loc, linewidth=2, label=label)

    # Mark time constants
    for i in range(1, 6):
        ax2.axvline(x=i*tau, color='gray', linestyle=':', alpha=0.5)
        if i <= 3:
            ax2.text(i*tau, ax2.get_ylim()[1]*0.95, f'{i}τ', ha='center', fontsize=8)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Temperature [°C]')
    ax2.set_title('Temperature vs Time')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_heat_sink_schematic(design, T_base):
    """Create detailed schematic of heat sink design"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Cross-section view (X-Z plane)
    ax1 = axes[0]

    # Base plate
    base_w = design['plate_width'] * 1000
    base_h = design['base_thickness'] * 1000
    base = Rectangle((0, 0), base_w, base_h, facecolor='lightsteelblue',
                      edgecolor='navy', linewidth=2)
    ax1.add_patch(base)

    # Fins
    n_fins = design['n_fins']
    fin_h = design['fin_height'] * 1000
    fin_t = design['fin_thickness'] * 1000
    spacing = (base_w - n_fins * fin_t) / (n_fins + 1)

    for i in range(n_fins):
        x = spacing + i * (fin_t + spacing)
        fin = Rectangle((x, base_h), fin_t, fin_h, facecolor='lightcoral',
                         edgecolor='darkred', linewidth=0.5)
        ax1.add_patch(fin)

    # Processor (heat source)
    proc = Rectangle((0, -3), base_w, 3, facecolor='gold',
                      edgecolor='orange', linewidth=2)
    ax1.add_patch(proc)
    ax1.text(base_w/2, -1.5, 'Processor (500W)', ha='center', va='center', fontsize=10)

    # Arrows showing heat flow
    ax1.annotate('', xy=(base_w/2, base_h + fin_h + 3),
                xytext=(base_w/2, -3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(base_w/2 + 3, (base_h + fin_h)/2, 'Heat flow', color='red',
             rotation=90, va='center')

    # Dimensions
    ax1.annotate('', xy=(0, base_h + fin_h + 5), xytext=(base_w, base_h + fin_h + 5),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax1.text(base_w/2, base_h + fin_h + 7, f'{base_w:.0f} mm', ha='center')

    ax1.annotate('', xy=(base_w + 3, base_h), xytext=(base_w + 3, base_h + fin_h),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax1.text(base_w + 5, base_h + fin_h/2, f'{fin_h:.1f} mm', va='center')

    ax1.set_xlim(-5, base_w + 15)
    ax1.set_ylim(-10, base_h + fin_h + 15)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Width [mm]')
    ax1.set_ylabel('Height [mm]')
    ax1.set_title('Cross-Section View')
    ax1.grid(True, alpha=0.3)

    # Right: Design summary
    ax2 = axes[1]
    ax2.axis('off')

    summary_text = f"""
    OPTIMAL HEAT SINK DESIGN
    ========================

    GEOMETRY
    --------
    Base plate: {design['plate_width']*1000:.0f} × {design['plate_depth']*1000:.0f} × {design['base_thickness']*1000:.1f} mm
    Number of fins: {design['n_fins']}
    Fin dimensions: {design['fin_width']*1000:.0f} × {design['fin_height']*1000:.2f} × {design['fin_thickness']*1000:.1f} mm
    Fin spacing: {spacing:.2f} mm
    Total height: {(design['base_thickness'] + design['fin_height'])*1000:.2f} mm

    MASS
    ----
    Base plate: {RHO * design['plate_width'] * design['plate_depth'] * design['base_thickness'] * 1000:.2f} g
    Fins (total): {RHO * design['n_fins'] * design['fin_width'] * design['fin_height'] * design['fin_thickness'] * 1000:.2f} g
    TOTAL: {(RHO * design['plate_width'] * design['plate_depth'] * design['base_thickness'] + RHO * design['n_fins'] * design['fin_width'] * design['fin_height'] * design['fin_thickness']) * 1000:.2f} g

    THERMAL PERFORMANCE
    -------------------
    Heat dissipated: {Q_TARGET:.0f} W
    Base temperature: {T_base:.2f} °C
    Max allowed: {T_MAX:.0f} °C
    Ambient: {T_INF:.0f} °C
    Margin: {T_MAX - T_base:.2f} °C

    MATERIAL: Aluminum
    ------------------
    k = {K} W/m-K
    ρ = {RHO} kg/m³
    h = {H} W/m²-K
    """

    ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top')

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TP3 - HEAT SINK DESIGN: VALIDATION AND RESULTS")
    print("="*60)

    design = OPTIMAL_DESIGN

    # 1. Code validation
    Q_anal, Q_num, T_anal, T_num, x_val = validate_fvm_code()
    fig1 = plot_validation_comparison(T_anal, T_num, x_val)
    fig1.savefig('validation_fvm.png', dpi=150, bbox_inches='tight')
    print("\nSaved: validation_fvm.png")

    # 2. Heat balance verification
    T_base, Q_total, info = verify_heat_balance(design)

    # 3. Transient simulation
    T_history, t_history, x_fin, tau = simulate_transient_startup(design, t_final=120.0)
    fig2 = plot_transient_evolution(T_history, t_history, x_fin, tau)
    fig2.savefig('transient_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: transient_evolution.png")

    # 4. Heat sink schematic
    fig3 = plot_heat_sink_schematic(design, T_base)
    fig3.savefig('heat_sink_design.png', dpi=150, bbox_inches='tight')
    print("Saved: heat_sink_design.png")

    # Final summary
    print("\n" + "="*60)
    print("FINAL DELIVERABLES SUMMARY")
    print("="*60)

    total_mass = (RHO * design['plate_width'] * design['plate_depth'] * design['base_thickness'] +
                  RHO * design['n_fins'] * design['fin_width'] * design['fin_height'] * design['fin_thickness'])

    print(f"""
    1. FIN GEOMETRY
       - Type: Flat rectangular fins
       - Height: {design['fin_height']*1000:.2f} mm
       - Thickness: {design['fin_thickness']*1000:.1f} mm
       - Width: {design['fin_width']*1000:.0f} mm (full plate depth)

    2. NUMBER OF FINS
       - {design['n_fins']} fins
       - Spacing: {(design['plate_width']*1000 - design['n_fins']*design['fin_thickness']*1000)/(design['n_fins']+1):.2f} mm

    3. HEAT SINK MASS
       - Total: {total_mass*1000:.2f} g ({total_mass:.4f} kg)
       - Base: {RHO * design['plate_width'] * design['plate_depth'] * design['base_thickness']*1000:.2f} g
       - Fins: {RHO * design['n_fins'] * design['fin_width'] * design['fin_height'] * design['fin_thickness']*1000:.2f} g

    4. THERMAL EVOLUTION
       - Time constant: τ ≈ {tau:.1f} s
       - 99% steady state: ~{5*tau:.0f} s
       - Base temperature at steady state: {T_base:.2f}°C

    5. VERIFICATION
       - Heat balance error: {abs(Q_TARGET - Q_total)/Q_TARGET*100:.2f}%
       - FVM validation error: {abs(Q_anal - Q_num)/Q_anal*100:.2f}%
    """)

    print("="*60)
    print("FILES GENERATED:")
    print("  - validation_fvm.png")
    print("  - transient_evolution.png")
    print("  - heat_sink_design.png")
    print("  - optimization_results.png")
    print("  - optimal_design.png")
    print("="*60)

    plt.show()
