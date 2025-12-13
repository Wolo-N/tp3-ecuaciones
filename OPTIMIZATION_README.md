# Heat Sink Fin Optimization

This README explains how to use the fin optimization code for the TP3 heat dissipator project.

## Files Overview

### Core Modules

1. **`fin_geometry.py`** - Q9 Mapped Grid Generator
   - Generates 2D finite volume grid using Q9 shape functions
   - Supports tapered fins (thick base → thin tip)
   - Functions:
     - `rebuild_grid_for_geometry()` - Update grid for new fin dimensions
     - `calculate_fin_mass()` - Calculate mass and volume
     - `build_fin_grid_2d()` - Get all geometric data structures

2. **`fin_2d_forward_euler.py`** - Thermal Solver
   - Forward Euler transient heat conduction solver
   - Functions:
     - `solve_transient()` - Solve 2D heat equation to steady state
     - `calculate_heat_dissipation()` - Calculate heat transfer rates

3. **`fin_optimizer.py`** - Optimization Module ⭐
   - **Main function**: `optimize_single_fin()`
   - Minimizes fin mass while meeting heat dissipation requirements
   - Uses scipy.optimize with SLSQP method
   - Constraints:
     - Heat dissipation ≥ target Q
     - Height ≤ 25mm (from specifications)
     - Base thickness ≥ tip thickness (taper constraint)

### Test/Example Scripts

4. **`test_optimization.py`** - Complete example with visualization
5. **`optimizer.py`** - Alternative grid search optimizer (your existing code)

---

## Quick Start: Optimize a Single Fin

```python
from fin_optimizer import optimize_single_fin

# Optimize a single fin to dissipate 20W
result = optimize_single_fin(
    q_target=20.0,           # Target heat dissipation (W)
    height_max=0.025,        # 25 mm max (from TP3 specs)
    thickness_min=0.0005,    # 0.5 mm min
    thickness_max=0.005,     # 5 mm max
    fin_length=0.050,        # 50 mm (processor width)
    k=205.0,                 # Aluminum thermal conductivity
    h=300.0,                 # Convection coefficient
    T_inf=45.0,              # Worst-case ambient (°C)
    T_base=90.0,             # CPU temperature limit (°C)
    verbose=True
)

# Access results
if result['success']:
    print(f"Optimal base thickness: {result['fin_thickness_base']*1000:.2f} mm")
    print(f"Optimal tip thickness:  {result['fin_thickness_tip']*1000:.2f} mm")
    print(f"Optimal height:         {result['fin_height']*1000:.2f} mm")
    print(f"Fin mass:               {result['fin_mass']*1000:.2f} g")
    print(f"Heat dissipated:        {result['q_convection']:.2f} W")
```

---

## Running the Examples

### Option 1: Run the test script (recommended)
```bash
python test_optimization.py
```
This will:
- Optimize a single fin for 20W heat dissipation
- Visualize the temperature distribution and geometry
- Calculate how many fins fit on the 50mm × 50mm base
- Estimate total heat sink mass

### Option 2: Run the optimizer directly
```bash
python fin_optimizer.py
```
This runs the basic example in the `if __name__ == "__main__"` block.

---

## Understanding the Results

The optimization returns a dictionary with these keys:

```python
{
    'success': bool,                    # Did optimization succeed?
    'fin_thickness_base': float,        # Base thickness (m)
    'fin_thickness_tip': float,         # Tip thickness (m)
    'fin_height': float,                # Height (m)
    'fin_length': float,                # Length (m)
    'fin_mass': float,                  # Single fin mass (kg)
    'fin_volume': float,                # Single fin volume (m³)
    'q_convection': float,              # Heat dissipated by convection (W)
    'q_base': float,                    # Heat entering through base (W)
    'q_balance': float,                 # Heat balance error (should be ~0)
    'message': str,                     # Optimization status message
    'T_field': ndarray,                 # Temperature at all nodes (°C)
    'coords': ndarray,                  # Node coordinates (m)
    'center_nodes': list,               # Center node indices
    'opt_result': OptimizeResult        # Full scipy optimization result
}
```

---

## Complete Heat Sink Design Workflow

To design the complete heat sink for **500W dissipation**:

### Step 1: Optimize Single Fin
Choose a target heat per fin (e.g., 15-25W) and run optimization:
```python
result = optimize_single_fin(q_target=20.0, ...)
```

### Step 2: Calculate Number of Fins
```python
base_width = 0.050      # 50 mm
fin_gap = 0.002         # 2 mm airflow gap
fin_thickness = result['fin_thickness_base']
pitch = fin_thickness + fin_gap
n_fins = int((base_width + fin_gap) / pitch)
```

### Step 3: Verify Total Heat Dissipation
```python
total_q = n_fins * result['q_convection']
print(f"Total heat dissipation: {total_q:.1f} W")
# Must be ≥ 500 W
```

### Step 4: Calculate Total Mass
```python
# Base plate
base_thickness = 0.005  # 5 mm
base_volume = 0.050 * 0.050 * base_thickness
base_mass = base_volume * 2700  # kg

# Total
total_mass = base_mass + n_fins * result['fin_mass']
print(f"Total heat sink mass: {total_mass*1000:.1f} g")
```

### Step 5: Iterate if Needed
If total heat < 500W:
- Increase `q_target` per fin
- Reduce `fin_gap` to fit more fins
- Try different fin shapes

---

## TP3 Requirements Checklist

From `TP3.pdf`, the design must:

- ✅ **Material**: Aluminum (k=205 W/m·K, ρ=2700 kg/m³, cp=900 J/kg·K)
- ✅ **Convection**: h = 300 W/m²·K
- ✅ **Max height**: 25 mm
- ✅ **Operating temp**: -20°C to 45°C (worst case: 45°C ambient)
- ✅ **Flat fins**: Base = 50mm
- ✅ **Heat removal**: 500 W total
- ✅ **CPU temp**: ≤ 90°C
- ✅ **Objective**: Minimize mass

### Deliverables
1. ✅ Fin geometry (thickness, height, shape)
2. ✅ Number of fins
3. ✅ Total heat sink mass
4. ✅ Thermal evolution to steady state (from solver)
5. Heat balance verification (convection = conduction)

---

## Advanced: Customizing the Optimization

### Change optimization method
```python
result = optimize_single_fin(
    q_target=20.0,
    method='trust-constr',  # Try: 'SLSQP', 'trust-constr', 'COBYLA'
    verbose=True
)
```

### Adjust bounds
```python
result = optimize_single_fin(
    q_target=20.0,
    height_max=0.020,        # Limit to 20mm instead of 25mm
    thickness_min=0.001,     # Thicker minimum (1mm)
    thickness_max=0.008,     # Allow thicker fins (8mm)
)
```

### Access full temperature field
```python
T = result['T_field']           # Temperature at all nodes
coords = result['coords']        # Node coordinates
centers = result['center_nodes'] # Center node indices

# Plot temperature profile
import matplotlib.pyplot as plt
plt.scatter(coords[centers, 1], T[centers], c=T[centers])
plt.xlabel('Height (m)')
plt.ylabel('Temperature (°C)')
plt.colorbar(label='Temperature (°C)')
plt.show()
```

---

## Grid Details

The Q9 grid uses:
- **11×11 nodes** (default) → **6×6 center nodes** (control volumes)
- **Q9 shape functions** for curved/tapered geometry
- **Finite Volume Method** for heat equation discretization
- **Forward Euler** explicit time integration

To change grid resolution, edit `nHeight` and `nWidth` in `fin_geometry.py` (must be odd numbers).

---

## Troubleshooting

### Optimization fails ("Positive directional derivative for linesearch")
- Try reducing `q_target` (less aggressive constraint)
- Increase `thickness_max` to give optimizer more freedom
- Change method to `'trust-constr'`

### Heat dissipation too low
- Increase fin height (up to 25mm limit)
- Increase fin thickness (more conduction)
- Check that T_inf = 45°C (worst case)

### Optimization very slow
- Reduce `t_final` in solve_transient (faster convergence)
- Set `verbose=False` to reduce print overhead
- Use coarser grid (edit `nHeight`, `nWidth` in fin_geometry.py)

---

## Contact & Support

For questions about:
- **Q9 grid generation** → See `fin_geometry.py` documentation
- **Thermal solver** → See `fin_2d_forward_euler.py` documentation
- **Optimization** → See `fin_optimizer.py` and scipy.optimize docs

---

## Next Steps

1. Run `python test_optimization.py` to see the example
2. Modify `q_target` to find optimal heat per fin
3. Calculate complete heat sink design (fins + base)
4. Verify 500W requirement is met
5. Create plots for presentation (temperature field, geometry, convergence)

Good luck with TP3! 🔥🎯
