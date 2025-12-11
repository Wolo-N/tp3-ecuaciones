# Heat Sink Fin Design — Development Plan

## Problem Understanding

```
Side view (2D grid = ONE FIN):        Top view (full heat sink):

    ▓▓  ← tip                         ═══════════════════════
    ▓▓                                ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║
    ▓▓  ← variable thickness          ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║  ← N fins
    ▓▓     along height               ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║     (50mm depth)
════════  ← base (bottom nodes)       ═══════════════════════
                                            50mm base
```

- **2D grid** = cross-section of **one fin**
- **fin_depth** = 50mm (extrusion into page, fixed by processor)
- **thickness[i]** = local fin thickness at each control volume (optimization variable)
- **Volume[i]** = Area[i] × fin_depth (for thermal mass)
- **Conduction area** between neighbors depends on thickness
- **500W** enters through base, distributed across N fins

## Existing Code
- `grid_matrices_2d_with_centers.py`: 2D grid, 36 CVs, areas, neighbors
- `fin_fvm_1d.py`: 1D FVM reference (boundary conditions, matrix assembly)

---

## Part 1: Geometry (`heat_sink_geometry.py`)

**Purpose**: Define single fin geometry with variable thickness

| Task | Description |
|------|-------------|
| Import grid | `structural_coords`, `center_nodes`, `neighbours`, `areas` from grid file |
| `fin_depth` | 0.05 m (50mm, fixed) |
| `thickness` | Array (n_centers,) — one value per CV, initially uniform ~2mm |
| `volumes` | `areas * fin_depth` (for thermal mass ρ·cp·V) |
| Material | `rho=2700`, `k=205`, `cp=900` (aluminum) |
| `compute_mass()` | `rho * sum(areas * thickness * fin_depth)` |

**Key insight**:
- `volumes` (for dT/dt term) uses `fin_depth`
- Conduction between neighbors uses `thickness` for cross-sectional area

---

## Part 2: Thermal Solver (`heat_sink_solver.py`)

**Purpose**: 2D transient FVM heat equation for one fin

| Task | Description |
|------|-------------|
| Import | Geometry from Part 1 |
| Identify boundaries | Bottom row = base, others = convection surfaces |
| Heat input | `Q_base = 500W / N_fins` applied to bottom CVs |
| Convection | `h=300 W/m²K`, `T_amb=45°C` on exposed surfaces |
| FVM discretization | Use `neighbours` array [C, L, R, U, D] |
| Time integration | Explicit Euler (simple), iterate until steady state |
| Outputs | `T[]` array, `Q_convected`, convergence history |

**FVM equation per CV:**
```
ρ·cp·V·(dT/dt) = Σ k·A_cond/dx·(T_nb - T) + h·A_conv·(T_amb - T) + Q_in
```

Where:
- `A_cond` = conduction area (depends on thickness & fin_depth)
- `A_conv` = convection area (exposed surface)
- `Q_in` = heat flux (only for base nodes)

---

## Part 3: Validation & Results (`heat_sink_results.py`)

**Purpose**: Verify solution and visualize

| Task | Description |
|------|-------------|
| Heat balance | Check `Q_in ≈ Q_convected` at steady state |
| Max temp check | Must be < 90°C |
| Plot T field | Color map on 2D grid |
| Plot T vs time | Transient evolution |
| Report mass | Single fin mass × N_fins |

---

## Files to Create

```
tp3/
├── grid_matrices_2d_with_centers.py  (existing)
├── heat_sink_geometry.py             (Part 1)
├── heat_sink_solver.py               (Part 2)
└── heat_sink_results.py              (Part 3)
```

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `fin_depth` | 50 mm | Fixed (processor width) |
| `thickness` | ~2 mm | Variable per CV, to optimize |
| `h` | 300 W/m²K | Convection coefficient |
| `T_amb` | 45°C | Worst case ambient |
| `T_max` | 90°C | Constraint |
| `Q_total` | 500 W | Split among N fins |
| `max_height` | 25 mm | Fin height constraint |
