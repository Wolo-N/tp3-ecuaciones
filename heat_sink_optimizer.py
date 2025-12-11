"""
Heat Sink Optimization Module

Optimizes the heat sink design to minimize mass while satisfying
the temperature constraint T_max <= 90C.

Decision Variables:
- N_fins: Number of fins (integer)
- W_base: Base half-width of fin [m]
- W_tip: Tip half-width of fin [m]
- H: Fin height [m]
- t: Fin thickness [m]

The Q9 control points are updated based on these parameters.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS
# =============================================================================

# Processor dimensions (fixed)
PROCESSOR_WIDTH = 0.05  # 50mm - this is the fin_depth (Z direction)
PROCESSOR_LENGTH = 0.05  # 50mm - fins arranged along this direction

# Material properties (Aluminum)
RHO = 2700.0  # kg/m3
K = 205.0  # W/m-K
CP = 900.0  # J/kg-K

# Thermal parameters
Q_TOTAL = 500.0  # Total heat [W]
H_CONV = 300.0  # Convection coefficient [W/m2-K]
T_AMB = 45.0  # Ambient temperature [C]
T_MAX = 90.0  # Maximum allowable temperature [C]

# Design bounds
MIN_FINS = 3
MAX_FINS = 25
MIN_WIDTH = 0.002  # 2mm minimum half-width
MAX_WIDTH = 0.025  # 25mm maximum half-width
MIN_HEIGHT = 0.010  # 10mm minimum height
MAX_HEIGHT = 0.050  # 50mm maximum height
MIN_THICKNESS = 0.001  # 1mm minimum thickness
MAX_THICKNESS = 0.010  # 10mm maximum thickness


# =============================================================================
# GEOMETRY FUNCTIONS
# =============================================================================

def update_q9_control_points(W_base, W_tip, H):
    """
    Generate Q9 control points for a symmetric tapered fin.

    Parameters
    ----------
    W_base : float
        Base half-width [m] (full base width = 2 * W_base)
    W_tip : float
        Tip half-width [m] (full tip width = 2 * W_tip)
    H : float
        Fin height [m]

    Returns
    -------
    ctrl : ndarray (9, 2)
        Q9 control points
    """
    # Y coordinates: base at -H/2, tip at +H/2
    y_base = -H / 2
    y_tip = H / 2
    y_mid = 0.0

    # X coordinates for mid-edge (linear interpolation)
    W_mid = (W_base + W_tip) / 2

    ctrl = np.array([
        [-W_base, y_base],    # N1 (-1,-1) bottom-left corner
        [ W_base, y_base],    # N2 ( 1,-1) bottom-right corner
        [ W_tip,  y_tip],     # N3 ( 1, 1) top-right corner
        [-W_tip,  y_tip],     # N4 (-1, 1) top-left corner
        [ 0.0,    y_base],    # N5 ( 0,-1) bottom-center
        [ W_mid,  y_mid],     # N6 ( 1, 0) right-center
        [ 0.0,    y_tip],     # N7 ( 0, 1) top-center
        [-W_mid,  y_mid],     # N8 (-1, 0) left-center
        [ 0.0,    y_mid]      # N9 ( 0, 0) center
    ])

    return ctrl


def compute_fin_mass(W_base, W_tip, H, thickness, fin_depth=PROCESSOR_WIDTH):
    """
    Compute mass of a single fin.

    The fin is a solid trapezoidal prism:
    - Cross-section in X-Y plane: trapezoid with base width 2*W_base, tip width 2*W_tip, height H
    - Extruded in Z direction by fin_depth
    - Additional material thickness t adds to the base width

    Volume = trapezoidal_area × fin_depth
           = (W_base + W_tip) × H × fin_depth

    The 'thickness' parameter represents additional material (e.g., thicker fins).
    We model this as: effective_W = W_base + thickness/2

    Parameters
    ----------
    W_base : float
        Base half-width [m]
    W_tip : float
        Tip half-width [m]
    H : float
        Fin height [m]
    thickness : float
        Additional fin thickness [m] (adds to base width)
    fin_depth : float
        Fin depth (extrusion) [m]

    Returns
    -------
    mass : float
        Fin mass [kg]
    """
    # Effective widths including thickness
    W_base_eff = W_base + thickness / 2
    W_tip_eff = W_tip + thickness / 2

    # Trapezoidal cross-section area
    # Area = (base_full_width + tip_full_width) / 2 × height
    #      = ((2×W_base_eff) + (2×W_tip_eff)) / 2 × H
    #      = (W_base_eff + W_tip_eff) × H
    cross_section_area = (W_base_eff + W_tip_eff) * H

    # Volume = cross_section_area × fin_depth
    volume = cross_section_area * fin_depth

    mass = RHO * volume

    return mass


def compute_total_mass(N_fins, W_base, W_tip, H, thickness):
    """
    Compute total heat sink mass.

    Parameters
    ----------
    N_fins : int
        Number of fins
    W_base, W_tip, H, thickness : float
        Fin geometry parameters [m]

    Returns
    -------
    total_mass : float
        Total heat sink mass [kg]
    """
    single_fin_mass = compute_fin_mass(W_base, W_tip, H, thickness)

    # Add base plate mass (simplified)
    # Base plate: PROCESSOR_LENGTH x PROCESSOR_WIDTH x base_thickness
    base_thickness = 0.003  # 3mm base plate
    base_mass = RHO * PROCESSOR_LENGTH * PROCESSOR_WIDTH * base_thickness

    total_mass = N_fins * single_fin_mass + base_mass

    return total_mass


# =============================================================================
# THERMAL SIMULATION (Simplified analytical model)
# =============================================================================

def estimate_Tmax_analytical(N_fins, W_base, W_tip, H, thickness):
    """
    Estimate maximum temperature using simplified analytical model.

    This uses fin efficiency equations for a quick estimate.
    Calibrated with FVM simulation results.

    Parameters
    ----------
    N_fins : int
        Number of fins
    W_base, W_tip, H, thickness : float
        Fin geometry parameters [m]

    Returns
    -------
    T_max : float
        Estimated maximum temperature [C]
    """
    # Heat per fin
    Q_fin = Q_TOTAL / N_fins

    # Average fin width
    W_avg = (W_base + W_tip) / 2

    # Fin perimeter (for convection)
    # Approximation: perimeter at average cross-section
    P = 2 * (2 * W_avg + PROCESSOR_WIDTH)  # front + back + sides

    # Fin cross-sectional area for conduction
    A_c = 2 * W_avg * PROCESSOR_WIDTH

    # Fin parameter m
    m = np.sqrt(H_CONV * P / (K * A_c))

    # Fin efficiency (assuming adiabatic tip)
    mL = m * H
    if mL > 0:
        eta_fin = np.tanh(mL) / mL
    else:
        eta_fin = 1.0

    # Convection area
    A_conv = P * H + 2 * W_avg * PROCESSOR_WIDTH  # sides + tip

    # Heat transfer
    # Q = h * A_conv * eta_fin * (T_base - T_amb)
    # T_base = T_amb + Q / (h * A_conv * eta_fin)

    # Effective thermal resistance
    if eta_fin > 0 and A_conv > 0:
        R_conv = 1 / (H_CONV * A_conv * eta_fin)
    else:
        R_conv = 1e6  # Very high resistance

    # Base temperature (maximum temperature for a fin)
    T_base = T_AMB + Q_fin * R_conv

    # Calibration factor (FVM shows ~86% of analytical estimate)
    # This accounts for 2D heat spreading effects not captured in 1D model
    CALIBRATION_FACTOR = 0.86
    T_base_calibrated = T_AMB + (T_base - T_AMB) * CALIBRATION_FACTOR

    return T_base_calibrated


def simulate_heat_sink(N_fins, W_base, W_tip, H, thickness, verbose=False):
    """
    Run full thermal simulation with updated geometry.

    Parameters
    ----------
    N_fins : int
        Number of fins
    W_base, W_tip, H, thickness : float
        Fin geometry parameters [m]
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Simulation results including T_max
    """
    import sys
    import importlib

    # Update the Q9 control points in the grid module
    ctrl = update_q9_control_points(W_base, W_tip, H)

    # We need to reload the modules with new control points
    # This is a bit hacky but necessary for optimization

    # Directly modify the grid module
    import grid_matrices_2d_with_centers as grid_module

    # Store original control points
    original_ctrl = grid_module.ctrl.copy()

    try:
        # Update control points
        grid_module.ctrl = ctrl

        # Recompute structural coordinates
        grid_module.structural_coords = grid_module.q9_interpolate_points(
            ctrl, grid_module.natural_coords
        )

        # Recompute areas
        for i, center in enumerate(grid_module.center_nodes):
            block = grid_module.blocks[center]
            grid_module.areas[i] = grid_module.compute_block_area(
                center, block, grid_module.structural_coords
            )

        # Reload geometry module to pick up changes
        import heat_sink_geometry as geom_module
        importlib.reload(geom_module)

        # Update N_fins and recompute Q_per_fin
        geom_module.N_fins = N_fins

        # Reload solver module
        import heat_sink_solver as solver_module
        importlib.reload(solver_module)

        # Update thickness array
        thickness_array = np.ones(geom_module.n_centers) * thickness

        # Run simulation
        solver, results = solver_module.solve_fin(thickness_array, verbose=verbose)

        return results

    finally:
        # Restore original control points
        grid_module.ctrl = original_ctrl
        grid_module.structural_coords = grid_module.q9_interpolate_points(
            original_ctrl, grid_module.natural_coords
        )
        for i, center in enumerate(grid_module.center_nodes):
            block = grid_module.blocks[center]
            grid_module.areas[i] = grid_module.compute_block_area(
                center, block, grid_module.structural_coords
            )


# =============================================================================
# OPTIMIZATION
# =============================================================================

def objective_function(x, use_full_simulation=False):
    """
    Objective function: minimize mass.

    Parameters
    ----------
    x : array
        [N_fins, W_base, W_tip, H, thickness]
    use_full_simulation : bool
        Use full FVM simulation (slow) or analytical estimate (fast)

    Returns
    -------
    mass : float
        Total heat sink mass [kg]
    """
    N_fins = int(round(x[0]))
    W_base = x[1]
    W_tip = x[2]
    H = x[3]
    thickness = x[4]

    return compute_total_mass(N_fins, W_base, W_tip, H, thickness)


def constraint_temperature(x, use_full_simulation=False):
    """
    Temperature constraint: T_max <= T_MAX (90C).

    Returns g(x) where g(x) >= 0 means constraint satisfied.

    Parameters
    ----------
    x : array
        [N_fins, W_base, W_tip, H, thickness]

    Returns
    -------
    g : float
        T_MAX - T_max (positive means satisfied)
    """
    N_fins = int(round(x[0]))
    W_base = x[1]
    W_tip = x[2]
    H = x[3]
    thickness = x[4]

    if use_full_simulation:
        results = simulate_heat_sink(N_fins, W_base, W_tip, H, thickness)
        T_max = results['T_max']
    else:
        T_max = estimate_Tmax_analytical(N_fins, W_base, W_tip, H, thickness)

    return T_MAX - T_max


def constraint_geometry(x):
    """
    Geometry constraints.

    Returns array of g(x) where g(x) >= 0 means satisfied.
    """
    N_fins = int(round(x[0]))
    W_base = x[1]
    W_tip = x[2]
    H = x[3]
    thickness = x[4]

    constraints = []

    # W_tip <= W_base (fin should taper or be straight, not expand)
    constraints.append(W_base - W_tip)

    # Fins must fit on processor
    # Each fin base width = 2 * W_base
    # Minimum spacing between fins = 2mm for airflow
    min_spacing = 0.002
    total_fin_width = N_fins * 2 * W_base
    total_spacing = (N_fins - 1) * min_spacing
    available_length = PROCESSOR_LENGTH
    constraints.append(available_length - total_fin_width - total_spacing)

    return np.array(constraints)


def optimize_heat_sink(use_full_simulation=False, verbose=True):
    """
    Run optimization to find minimum mass design.

    Parameters
    ----------
    use_full_simulation : bool
        Use full FVM simulation (slow but accurate) or
        analytical estimate (fast but approximate)
    verbose : bool
        Print progress

    Returns
    -------
    result : dict
        Optimization results
    """
    # Bounds: [N_fins, W_base, W_tip, H, thickness]
    bounds = [
        (MIN_FINS, MAX_FINS),           # N_fins
        (MIN_WIDTH, MAX_WIDTH),         # W_base
        (MIN_WIDTH, MAX_WIDTH),         # W_tip
        (MIN_HEIGHT, MAX_HEIGHT),       # H
        (MIN_THICKNESS, MAX_THICKNESS)  # thickness
    ]

    # Initial guess
    x0 = np.array([10, 0.015, 0.010, 0.025, 0.003])

    if verbose:
        print("=" * 70)
        print("HEAT SINK OPTIMIZATION")
        print("=" * 70)
        print(f"\nObjective: Minimize total mass")
        print(f"Constraint: T_max <= {T_MAX} C")
        print(f"\nDecision variables:")
        print(f"  - N_fins:    [{MIN_FINS}, {MAX_FINS}]")
        print(f"  - W_base:    [{MIN_WIDTH*1000:.1f}, {MAX_WIDTH*1000:.1f}] mm")
        print(f"  - W_tip:     [{MIN_WIDTH*1000:.1f}, {MAX_WIDTH*1000:.1f}] mm")
        print(f"  - H:         [{MIN_HEIGHT*1000:.1f}, {MAX_HEIGHT*1000:.1f}] mm")
        print(f"  - thickness: [{MIN_THICKNESS*1000:.1f}, {MAX_THICKNESS*1000:.1f}] mm")

    # Use differential evolution for global optimization
    if verbose:
        print(f"\nRunning differential evolution optimization...")

    def combined_objective(x):
        """Objective with penalty for constraint violations."""
        N_fins = int(round(x[0]))
        W_base = x[1]
        W_tip = x[2]
        H = x[3]
        thickness = x[4]

        # Enforce W_tip <= W_base (conventional tapered fin)
        if W_tip > W_base:
            return 1e6  # Infeasible

        mass = objective_function(x, use_full_simulation)

        # Temperature constraint
        temp_margin = constraint_temperature(x, use_full_simulation)
        if temp_margin < 0:
            # Penalty for constraint violation
            mass += 1000 * abs(temp_margin)

        # Geometry constraints
        geom_margins = constraint_geometry(x)
        for g in geom_margins:
            if g < 0:
                mass += 1000 * abs(g)

        return mass

    # Callback for progress
    iteration_count = [0]
    best_mass = [float('inf')]

    def callback(xk, convergence=None):
        iteration_count[0] += 1
        mass = objective_function(xk, use_full_simulation)
        if mass < best_mass[0]:
            best_mass[0] = mass
        if verbose and iteration_count[0] % 10 == 0:
            print(f"  Iteration {iteration_count[0]}: best mass = {best_mass[0]*1000:.2f} g")

    result = differential_evolution(
        combined_objective,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        callback=callback,
        disp=verbose,
        polish=True
    )

    # Extract results
    x_opt = result.x
    N_fins_opt = int(round(x_opt[0]))
    W_base_opt = x_opt[1]
    W_tip_opt = x_opt[2]
    H_opt = x_opt[3]
    thickness_opt = x_opt[4]

    mass_opt = compute_total_mass(N_fins_opt, W_base_opt, W_tip_opt, H_opt, thickness_opt)
    T_max_opt = estimate_Tmax_analytical(N_fins_opt, W_base_opt, W_tip_opt, H_opt, thickness_opt)

    opt_result = {
        'N_fins': N_fins_opt,
        'W_base': W_base_opt,
        'W_tip': W_tip_opt,
        'H': H_opt,
        'thickness': thickness_opt,
        'mass': mass_opt,
        'T_max_estimated': T_max_opt,
        'success': result.success,
        'message': result.message,
        'raw_result': result
    }

    if verbose:
        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS")
        print("=" * 70)
        print(f"\nOptimal Design:")
        print(f"  - Number of fins:   {N_fins_opt}")
        print(f"  - Base half-width:  {W_base_opt*1000:.2f} mm (full: {2*W_base_opt*1000:.2f} mm)")
        print(f"  - Tip half-width:   {W_tip_opt*1000:.2f} mm (full: {2*W_tip_opt*1000:.2f} mm)")
        print(f"  - Fin height:       {H_opt*1000:.2f} mm")
        print(f"  - Fin thickness:    {thickness_opt*1000:.2f} mm")
        print(f"\nPerformance:")
        print(f"  - Total mass:       {mass_opt*1000:.2f} g")
        print(f"  - T_max (estimated):{T_max_opt:.2f} C")
        print(f"  - Constraint:       {'SATISFIED' if T_max_opt <= T_MAX else 'VIOLATED'}")
        print("=" * 70)

    return opt_result


def validate_optimal_design(opt_result, verbose=True):
    """
    Validate optimal design using full FVM simulation.

    Parameters
    ----------
    opt_result : dict
        Results from optimize_heat_sink()
    verbose : bool
        Print progress

    Returns
    -------
    validation : dict
        Validation results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATING OPTIMAL DESIGN WITH FVM SIMULATION")
        print("=" * 70)

    # Run full simulation
    results = simulate_heat_sink(
        opt_result['N_fins'],
        opt_result['W_base'],
        opt_result['W_tip'],
        opt_result['H'],
        opt_result['thickness'],
        verbose=verbose
    )

    validation = {
        'T_max_simulated': results['T_max'],
        'T_max_estimated': opt_result['T_max_estimated'],
        'constraint_satisfied': results['T_max'] <= T_MAX,
        'estimation_error': abs(results['T_max'] - opt_result['T_max_estimated']),
        'full_results': results
    }

    if verbose:
        print(f"\nValidation Results:")
        print(f"  - T_max (FVM):      {results['T_max']:.2f} C")
        print(f"  - T_max (estimate): {opt_result['T_max_estimated']:.2f} C")
        print(f"  - Estimation error: {validation['estimation_error']:.2f} C")
        print(f"  - Constraint:       {'SATISFIED' if validation['constraint_satisfied'] else 'VIOLATED'}")

    return validation


def parametric_study(variable='N_fins', values=None, verbose=True):
    """
    Perform parametric study varying one parameter.

    Parameters
    ----------
    variable : str
        Variable to vary: 'N_fins', 'W_base', 'H', 'thickness'
    values : array, optional
        Values to test
    verbose : bool
        Print progress

    Returns
    -------
    study_results : dict
        Results for each parameter value
    """
    # Default values
    base_design = {
        'N_fins': 10,
        'W_base': 0.015,
        'W_tip': 0.010,
        'H': 0.025,
        'thickness': 0.003
    }

    if values is None:
        if variable == 'N_fins':
            values = np.arange(3, 21, 1)
        elif variable == 'W_base':
            values = np.linspace(0.005, 0.020, 10)
        elif variable == 'H':
            values = np.linspace(0.010, 0.050, 10)
        elif variable == 'thickness':
            values = np.linspace(0.001, 0.008, 10)
        else:
            raise ValueError(f"Unknown variable: {variable}")

    results = {
        'variable': variable,
        'values': values,
        'mass': [],
        'T_max': []
    }

    if verbose:
        print(f"\nParametric study: varying {variable}")
        print("-" * 40)

    for val in values:
        design = base_design.copy()
        design[variable] = val

        # Ensure W_tip <= W_base
        if variable == 'W_base' and design['W_tip'] > design['W_base']:
            design['W_tip'] = design['W_base'] * 0.7

        mass = compute_total_mass(
            design['N_fins'],
            design['W_base'],
            design['W_tip'],
            design['H'],
            design['thickness']
        )

        T_max = estimate_Tmax_analytical(
            design['N_fins'],
            design['W_base'],
            design['W_tip'],
            design['H'],
            design['thickness']
        )

        results['mass'].append(mass)
        results['T_max'].append(T_max)

        if verbose:
            if variable == 'N_fins':
                print(f"  {variable}={val:2d}: mass={mass*1000:.2f}g, T_max={T_max:.1f}C")
            else:
                print(f"  {variable}={val*1000:.2f}mm: mass={mass*1000:.2f}g, T_max={T_max:.1f}C")

    results['mass'] = np.array(results['mass'])
    results['T_max'] = np.array(results['T_max'])

    return results


def plot_parametric_study(study_results, save_path=None):
    """
    Plot parametric study results.

    Parameters
    ----------
    study_results : dict
        Results from parametric_study()
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    variable = study_results['variable']
    values = study_results['values']

    if variable != 'N_fins':
        values = values * 1000  # Convert to mm
        xlabel = f'{variable} [mm]'
    else:
        xlabel = variable

    # Mass plot
    ax1.plot(values, study_results['mass'] * 1000, 'b-o', linewidth=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Total Mass [g]')
    ax1.set_title(f'Mass vs {variable}')
    ax1.grid(True, alpha=0.3)

    # Temperature plot
    ax2.plot(values, study_results['T_max'], 'r-o', linewidth=2)
    ax2.axhline(y=T_MAX, color='red', linestyle='--', label=f'T_max limit = {T_MAX}C')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Maximum Temperature [C]')
    ax2.set_title(f'T_max vs {variable}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run parametric studies first
    print("Running parametric studies...")

    study_Nfins = parametric_study('N_fins')
    study_H = parametric_study('H')

    # Plot studies
    import matplotlib
    matplotlib.use('Agg')

    fig1 = plot_parametric_study(study_Nfins, 'parametric_Nfins.png')
    fig2 = plot_parametric_study(study_H, 'parametric_H.png')

    print("\nParametric study plots saved!")

    # Run optimization
    opt_result = optimize_heat_sink(use_full_simulation=False, verbose=True)

    print("\nOptimization complete!")
