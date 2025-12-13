"""
Fin Optimization Module

Simple optimization functions to find optimal fin geometries that minimize mass
while meeting heat dissipation requirements.

Uses:
- Q9 mapped grid from fin_geometry.py
- Forward Euler thermal solver from fin_2d_forward_euler.py
- scipy.optimize for constrained optimization
"""

import numpy as np
from scipy.optimize import minimize

import fin_geometry as geom
from fin_2d_forward_euler import solve_transient, calculate_heat_dissipation


# =============================================================================
# Helper functions for evaluation
# =============================================================================

def evaluate_fin_geometry(thickness_base, thickness_tip, height,
                          fin_length, rho, k, h, cp, T_inf, T_base):
    """
    Evaluate a fin geometry: rebuild grid, solve thermal, return mass and heat.

    Parameters
    ----------
    thickness_base : float
        Fin thickness at base (m)
    thickness_tip : float
        Fin thickness at tip (m)
    height : float
        Fin height (m)
    fin_length : float
        Fin length (m)
    rho : float
        Density (kg/m³)
    k : float
        Thermal conductivity (W/m·K)
    h : float
        Convection coefficient (W/m²·K)
    cp : float
        Specific heat (J/kg·K)
    T_inf : float
        Ambient temperature (°C)
    T_base : float
        Base temperature (°C)

    Returns
    -------
    dict with keys:
        'mass' : float (kg)
        'volume' : float (m³)
        'q_convection' : float (W)
        'q_base' : float (W)
        'T' : ndarray
        'coords' : ndarray
        'center_nodes' : list
    """
    # Rebuild grid with new geometry
    geom.rebuild_grid_for_geometry(
        new_fin_thickness=thickness_base,
        new_fin_height=height,
        new_fin_length=fin_length,
        new_fin_thickness_tip=thickness_tip
    )

    # Calculate mass
    mass, volume = geom.calculate_fin_mass(rho)

    # Solve thermal problem
    T, coords, center_nodes = solve_transient(
        k=k, h=h, rho=rho, cp=cp,
        T_inf=T_inf, T_base=T_base,
        dt=0.01, t_final=5.0, tol=1e-6
    )

    # Calculate heat dissipation
    q_dict = calculate_heat_dissipation(
        T, coords, center_nodes, geom.neighbours_dict,
        geom.face_areas, geom.center_distances, geom.boundary_areas,
        k, h, T_inf
    )

    return {
        'mass': mass,
        'volume': volume,
        'q_convection': q_dict['q_convection'],
        'q_base': q_dict['q_base'],
        'q_balance': q_dict['q_balance'],
        'T': T,
        'coords': coords,
        'center_nodes': center_nodes
    }


def compute_objective(x, params):
    """
    Objective function: minimize fin mass.

    Parameters
    ----------
    x : array
        Design variables [thickness_base, thickness_tip, height]
    params : dict
        Fixed parameters (fin_length, rho, k, h, cp, T_inf, T_base, verbose, eval_count)

    Returns
    -------
    mass : float
        Fin mass (kg)
    """
    thickness_base, thickness_tip, height = x

    # Rebuild geometry and calculate mass only
    geom.rebuild_grid_for_geometry(
        new_fin_thickness=thickness_base,
        new_fin_height=height,
        new_fin_length=params['fin_length'],
        new_fin_thickness_tip=thickness_tip
    )

    mass, _ = geom.calculate_fin_mass(params['rho'])

    # Update counter and print progress
    params['eval_count'][0] += 1
    if params['verbose'] and params['eval_count'][0] % 10 == 0:
        print(f"Eval {params['eval_count'][0]}: mass={mass*1000:.2f}g, "
              f"t_base={thickness_base*1000:.2f}mm, "
              f"t_tip={thickness_tip*1000:.2f}mm, "
              f"h={height*1000:.1f}mm")

    return mass


def compute_heat_constraint(x, params):
    """
    Heat constraint: q_convection >= q_target.

    For scipy.optimize inequality constraint, we return: q_conv - q_target
    This must be >= 0 for the constraint to be satisfied.

    Parameters
    ----------
    x : array
        Design variables [thickness_base, thickness_tip, height]
    params : dict
        Fixed parameters including q_target

    Returns
    -------
    constraint_value : float
        q_convection - q_target (must be >= 0)
    """
    thickness_base, thickness_tip, height = x

    # Evaluate geometry completely
    result = evaluate_fin_geometry(
        thickness_base=thickness_base,
        thickness_tip=thickness_tip,
        height=height,
        fin_length=params['fin_length'],
        rho=params['rho'],
        k=params['k'],
        h=params['h'],
        cp=params['cp'],
        T_inf=params['T_inf'],
        T_base=params['T_base']
    )

    q_conv = result['q_convection']

    if params['verbose']:
        print(f"  -> q_conv={q_conv:.2f}W (target={params['q_target']:.2f}W)")

    return q_conv - params['q_target']


def compute_taper_constraint(x):
    """
    Taper constraint: base thickness >= tip thickness.

    Parameters
    ----------
    x : array
        Design variables [thickness_base, thickness_tip, height]

    Returns
    -------
    constraint_value : float
        thickness_base - thickness_tip (must be >= 0)
    """
    thickness_base, thickness_tip, _ = x
    return thickness_base - thickness_tip


# =============================================================================
# Main optimization function
# =============================================================================

def optimize_single_fin(q_target,
                        height_max=0.025,
                        thickness_min=0.0005,
                        thickness_max=0.005,
                        fin_length=0.050,
                        k=205.0,
                        h=300.0,
                        rho=2700.0,
                        cp=900.0,
                        T_inf=45.0,
                        T_base=90.0,
                        method='SLSQP',
                        verbose=True):
    """
    Optimize a single fin to minimize mass while meeting heat dissipation target.

    Uses scipy.optimize to find optimal fin thickness (base and tip) and height
    that minimizes fin mass while dissipating at least q_target watts.

    The fin shape is created using the Q9 mapped grid (tapered or rectangular).

    Parameters
    ----------
    q_target : float
        Target heat dissipation per fin (W)
    height_max : float
        Maximum fin height (m), default 0.025 m = 25 mm
    thickness_min : float
        Minimum fin thickness (m), default 0.0005 m = 0.5 mm
    thickness_max : float
        Maximum fin thickness (m), default 0.005 m = 5 mm
    fin_length : float
        Fin length (m), default 0.050 m = 50 mm
    k : float
        Thermal conductivity (W/m·K), default 205 for aluminum
    h : float
        Convection coefficient (W/m²·K), default 300
    rho : float
        Density (kg/m³), default 2700 for aluminum
    cp : float
        Specific heat (J/kg·K), default 900 for aluminum
    T_inf : float
        Ambient temperature (°C), default 45
    T_base : float
        Base temperature (°C), default 90
    method : str
        Optimization method, default 'SLSQP'
    verbose : bool
        Print optimization progress

    Returns
    -------
    result : dict
        Optimization result with keys:
            'success' : bool
            'fin_thickness_base' : float (m)
            'fin_thickness_tip' : float (m)
            'fin_height' : float (m)
            'fin_mass' : float (kg)
            'fin_volume' : float (m³)
            'q_convection' : float (W)
            'q_base' : float (W)
            'q_balance' : float (W)
            'message' : str
            'T_field' : ndarray
            'coords' : ndarray
            'center_nodes' : list
    """
    # Pack parameters into dict for passing to objective/constraint functions
    params = {
        'q_target': q_target,
        'fin_length': fin_length,
        'k': k,
        'h': h,
        'rho': rho,
        'cp': cp,
        'T_inf': T_inf,
        'T_base': T_base,
        'verbose': verbose,
        'eval_count': [0]  # Mutable counter
    }

    # Initial guess: mid-range values
    x0 = np.array([
        (thickness_min + thickness_max) / 2,  # base thickness
        thickness_min,                         # tip thickness (start thin)
        height_max * 0.8                       # height (start near max)
    ])

    # Bounds for design variables
    bounds = [
        (thickness_min, thickness_max),  # base thickness
        (thickness_min, thickness_max),  # tip thickness
        (0.010, height_max)              # height (min 10mm)
    ]

    # Constraints
    constraints = [
        {
            'type': 'ineq',
            'fun': compute_heat_constraint,
            'args': (params,)
        },
        {
            'type': 'ineq',
            'fun': compute_taper_constraint
        }
    ]

    if verbose:
        print(f"\n{'='*70}")
        print(f"STARTING OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Target: Minimize mass with Q >= {q_target:.2f} W")
        print(f"Initial guess:")
        print(f"  t_base = {x0[0]*1000:.2f} mm")
        print(f"  t_tip  = {x0[1]*1000:.2f} mm")
        print(f"  height = {x0[2]*1000:.1f} mm")
        print(f"{'='*70}\n")

    # Run optimization
    opt_result = minimize(
        fun=compute_objective,
        x0=x0,
        args=(params,),
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'disp': verbose, 'maxiter': 100}
    )

    # Extract optimal design
    thickness_base_opt = opt_result.x[0]
    thickness_tip_opt = opt_result.x[1]
    height_opt = opt_result.x[2]

    # Evaluate final optimal design
    final_result = evaluate_fin_geometry(
        thickness_base=thickness_base_opt,
        thickness_tip=thickness_tip_opt,
        height=height_opt,
        fin_length=fin_length,
        rho=rho,
        k=k,
        h=h,
        cp=cp,
        T_inf=T_inf,
        T_base=T_base
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Status: {'SUCCESS' if opt_result.success else 'FAILED'}")
        print(f"Message: {opt_result.message}")
        print(f"\nOptimal single fin geometry:")
        print(f"  Base thickness: {thickness_base_opt*1000:.3f} mm")
        print(f"  Tip thickness:  {thickness_tip_opt*1000:.3f} mm")
        print(f"  Height:         {height_opt*1000:.2f} mm")
        print(f"  Length:         {fin_length*1000:.1f} mm")
        print(f"\nPerformance:")
        print(f"  Mass:           {final_result['mass']*1000:.3f} g")
        print(f"  Volume:         {final_result['volume']*1e9:.3f} mm³")
        print(f"  Q_convection:   {final_result['q_convection']:.2f} W")
        print(f"  Q_base:         {final_result['q_base']:.2f} W")
        print(f"  Heat balance:   {final_result['q_balance']:.4f} W")
        print(f"  Target Q:       {q_target:.2f} W")
        print(f"  Margin:         {final_result['q_convection'] - q_target:.2f} W")
        print(f"{'='*70}\n")

    return {
        'success': opt_result.success,
        'fin_thickness_base': thickness_base_opt,
        'fin_thickness_tip': thickness_tip_opt,
        'fin_height': height_opt,
        'fin_length': fin_length,
        'fin_mass': final_result['mass'],
        'fin_volume': final_result['volume'],
        'q_convection': final_result['q_convection'],
        'q_base': final_result['q_base'],
        'q_balance': final_result['q_balance'],
        'message': opt_result.message,
        'T_field': final_result['T'],
        'coords': final_result['coords'],
        'center_nodes': final_result['center_nodes'],
        'opt_result': opt_result
    }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SINGLE FIN OPTIMIZATION EXAMPLE")
    print("="*70)

    # Target heat per fin (example: 20W per fin -> would need ~25 fins for 500W)
    q_per_fin = 20.0  # W

    # Run optimization
    result = optimize_single_fin(
        q_target=q_per_fin,
        height_max=0.025,      # 25 mm max height (per specifications)
        thickness_min=0.0005,  # 0.5 mm min thickness
        thickness_max=0.005,   # 5 mm max thickness
        fin_length=0.050,      # 50 mm (processor width)
        k=205.0,               # Aluminum thermal conductivity
        h=300.0,               # Convection coefficient (W/m²·K)
        rho=2700.0,            # Aluminum density (kg/m³)
        cp=900.0,              # Aluminum specific heat (J/kg·K)
        T_inf=45.0,            # Worst case ambient temperature (°C)
        T_base=90.0,           # CPU temperature limit (°C)
        method='SLSQP',
        verbose=True
    )

    if result['success']:
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Calculate how many fins fit on the 50mm x 50mm base")
        print("2. Verify total heat dissipation meets 500W requirement")
        print("3. Calculate total heat sink mass (fins + base plate)")
        print("4. Iterate if needed to meet all constraints")
        print("="*70 + "\n")
