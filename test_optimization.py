"""
Test script for single fin optimization using Q9 grid.

This script demonstrates how to optimize a single fin geometry to:
1. Minimize material (mass/volume)
2. Meet heat dissipation requirements
3. Stay within geometric constraints (25mm max height)

The optimization uses:
- Q9 mapped grid from fin_geometry.py
- Forward Euler transient solver from fin_2d_forward_euler.py
- scipy.optimize for constrained optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from fin_optimizer import optimize_single_fin
from fin_geometry import rebuild_grid_for_geometry, structural_coords, center_nodes

# =============================================================================
# PROBLEM PARAMETERS (from TP3.pdf)
# =============================================================================
# Total heat to dissipate: 500 W
# Processor area: 50mm x 50mm
# Max CPU temperature: 90°C
# Operating temperature range: -20°C to 45°C (worst case: 45°C ambient)
# Max heat sink height: 25 mm
# Material: Aluminum
# Convection coefficient: h = 300 W/m²·K

# Aluminum properties
k_al = 205.0      # W/m·K - thermal conductivity
rho_al = 2700.0   # kg/m³ - density
cp_al = 900.0     # J/kg·K - specific heat

# Thermal conditions
h = 300.0         # W/m²·K - convection coefficient
T_base = 90.0     # °C - CPU max temperature
T_inf = 45.0      # °C - worst case ambient temperature

# Geometric constraints
height_max = 0.025      # m (25 mm)
fin_length = 0.050      # m (50 mm processor width)

# =============================================================================
# EXAMPLE 1: Optimize for a single fin dissipating 20W
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: Single Fin Optimization for Q = 20W")
print("="*80)

result = optimize_single_fin(
    q_target=20.0,              # Target 20W per fin
    height_max=height_max,
    thickness_min=0.0005,       # 0.5 mm min
    thickness_max=0.005,        # 5 mm max
    fin_length=fin_length,
    k=k_al,
    h=h,
    rho=rho_al,
    cp=cp_al,
    T_inf=T_inf,
    T_base=T_base,
    method='SLSQP',
    verbose=True
)

# =============================================================================
# Visualize the optimized fin geometry
# =============================================================================
if result['success']:
    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)

    # Rebuild geometry with optimal parameters
    rebuild_grid_for_geometry(
        new_fin_thickness=result['fin_thickness_base'],
        new_fin_height=result['fin_height'],
        new_fin_length=result['fin_length'],
        new_fin_thickness_tip=result['fin_thickness_tip']
    )

    # Plot the optimized fin shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Temperature field
    T_field = result['T_field']
    coords = result['coords']
    centers = result['center_nodes']

    # Exaggerate x-axis for visualization (fin is very thin)
    x_scale = 10.0
    coords_plot = coords.copy()
    coords_plot[:, 0] *= x_scale

    scatter = ax1.scatter(coords_plot[centers, 0], coords_plot[centers, 1],
                         c=T_field[centers], cmap='hot', s=100, edgecolors='k')
    ax1.set_xlabel(f'Width (x{x_scale} exaggerated) [m]')
    ax1.set_ylabel('Height [m]')
    ax1.set_title('Optimized Fin Temperature Distribution')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Temperature [°C]')

    # Plot 2: Fin geometry profile
    ax2.plot(coords_plot[:, 0], coords_plot[:, 1], 'b.', markersize=2, alpha=0.5, label='Grid nodes')
    ax2.plot(coords_plot[centers, 0], coords_plot[centers, 1], 'ro', markersize=6, label='Center nodes')
    ax2.set_xlabel(f'Width (x{x_scale} exaggerated) [m]')
    ax2.set_ylabel('Height [m]')
    ax2.set_title('Optimized Fin Geometry (Q9 Grid)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('optimized_fin_geometry.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'optimized_fin_geometry.png'")
    plt.show()

    # =============================================================================
    # Calculate how many fins fit on the base
    # =============================================================================
    print("\n" + "="*80)
    print("FIN ARRAY ANALYSIS")
    print("="*80)

    base_width = 0.050  # m (50 mm)
    fin_gap = 0.002     # m (2 mm gap for airflow)

    fin_thickness_base = result['fin_thickness_base']
    pitch = fin_thickness_base + fin_gap
    n_fins = int(np.floor((base_width + fin_gap) / pitch))

    print(f"Base width: {base_width*1000:.1f} mm")
    print(f"Fin base thickness: {fin_thickness_base*1000:.3f} mm")
    print(f"Fin gap (for airflow): {fin_gap*1000:.1f} mm")
    print(f"Pitch (fin + gap): {pitch*1000:.3f} mm")
    print(f"Number of fins that fit: {n_fins}")

    total_q_fins = n_fins * result['q_convection']
    print(f"\nHeat dissipation:")
    print(f"  Per fin: {result['q_convection']:.2f} W")
    print(f"  Total from {n_fins} fins: {total_q_fins:.2f} W")
    print(f"  Target: 500 W")
    print(f"  Shortfall: {500 - total_q_fins:.2f} W")

    # Calculate total mass
    base_thickness = 0.005  # m (5 mm base plate)
    base_volume = base_width * fin_length * base_thickness
    base_mass = base_volume * rho_al
    fins_mass = n_fins * result['fin_mass']
    total_mass = base_mass + fins_mass

    print(f"\nMass analysis:")
    print(f"  Single fin mass: {result['fin_mass']*1000:.3f} g")
    print(f"  Total fins mass: {fins_mass*1000:.2f} g")
    print(f"  Base plate mass: {base_mass*1000:.2f} g")
    print(f"  TOTAL HEAT SINK MASS: {total_mass*1000:.2f} g ({total_mass:.4f} kg)")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    if total_q_fins >= 500:
        print("✓ Heat dissipation requirement MET!")
    else:
        print("✗ Heat dissipation requirement NOT MET")
        print(f"  Consider:")
        print(f"    - Increase target Q per fin (current: {result['q_convection']:.2f}W)")
        print(f"    - Reduce fin gap to fit more fins")
        print(f"    - Optimize for different fin shape")

    print("="*80 + "\n")

else:
    print("\n" + "="*80)
    print("OPTIMIZATION FAILED")
    print("="*80)
    print(f"Message: {result['message']}")
    print("Try adjusting constraints or initial guess.")
    print("="*80 + "\n")
