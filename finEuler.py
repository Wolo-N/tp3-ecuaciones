import numpy as np

# Import your geometry builder
from finGeometry import build_fin_grid_2d

def stable_timestep(center_nodes, neighbours_dict, face_areas, center_distances,
                    boundary_areas, volumes_dict, rho, cp, k, h, safety=0.5):
    """
    Estimate a stable explicit timestep using a lumped-capacitance bound:
        dt_max = safety * (rho*cp*V) / (sum(k*A/d) + h*A_surf)
    Returns the minimum dt_max over all control volumes.
    """
    dt_candidates = []
    for c in center_nodes:
        V = volumes_dict[c]
        cond_sum = 0.0
        for j in neighbours_dict[c]:
            if j is None or j == c:
                continue
            A = face_areas[(c, j)]
            d = center_distances[(c, j)]
            cond_sum += k * A / d
        conv_sum = h * boundary_areas[c]
        denom = cond_sum + conv_sum
        if denom <= 0:
            continue
        dt_max = safety * (rho * cp * V) / denom
        dt_candidates.append(dt_max)
    return min(dt_candidates) if dt_candidates else None

# ------------------------------------------------------------
# Forward Euler update for one control volume
# ------------------------------------------------------------
def update_one_cell(c, T, dt, k, h, rho, cp, T_inf,
                    neighbours_dict, face_areas, center_distances,
                    boundary_areas, volumes_dict):

    Q_cond = 0.0

    # --- conduction term ---
    for j in neighbours_dict[c]:
        if j is None or j == c:
            continue
        A = face_areas[(c, j)]
        d = center_distances[(c, j)]
        Q_cond += k * A * (T[j] - T[c]) / d

    # --- convection term ---
    A_surf = boundary_areas[c]
    Q_conv = -h * A_surf * (T[c] - T_inf)

    # --- explicit update ---
    V = volumes_dict[c]
    T_new = T[c] + dt * (Q_cond + Q_conv) / (rho * cp * V)

    return T_new


# ------------------------------------------------------------
# One Forward Euler step for all control volumes
# ------------------------------------------------------------
def forward_euler_step(T, dt, k, h, rho, cp, T_inf,
                       center_nodes, neighbours_dict,
                       face_areas, center_distances,
                       boundary_areas, volumes_dict,
                       bottom_cells, T_base):

    T_new = T.copy()

    # --- update only center nodes ---
    for c in center_nodes:
        T_new[c] = update_one_cell(c, T, dt, k, h, rho, cp, T_inf,
                                   neighbours_dict, face_areas,
                                   center_distances, boundary_areas,
                                   volumes_dict)

    # --- enforce Dirichlet BC at CPU base ---
    for c in bottom_cells:
        T_new[c] = T_base

    return T_new


# ------------------------------------------------------------
# Transient solver
# ------------------------------------------------------------
def solve_transient(k, h, rho, cp, T_inf, T_base,
                    dt=0.01, t_final=5.0, tol=1e-6,
                    save_history=False, save_interval=0.1):

    # --- load geometry ---
    (coords,
     center_nodes,
     neighbours_dict,
     areas,
     volumes,
     blocks,
     center_distances,
     face_lengths,
     face_areas,
     boundary_areas) = build_fin_grid_2d()

    # Create temperature array for ALL nodes
    T = np.ones(len(coords)) * T_inf

    # Identify bottom boundary CVs via geometry (lowest y among center nodes)
    y_coords = coords[center_nodes, 1]
    y_min = y_coords.min()
    tol_geom = 1e-12 + 1e-6 * abs(y_min)
    bottom_cells = [c for c in center_nodes if coords[c, 1] <= y_min + tol_geom]

    # Apply base temperature initially
    for c in bottom_cells:
        T[c] = T_base

    # --- stability check and adjust dt if needed ---
    dt_stable = stable_timestep(center_nodes, neighbours_dict, face_areas,
                                center_distances, boundary_areas, volumes,
                                rho, cp, k, h, safety=0.5)
    if dt_stable is not None and dt > dt_stable:
        print(f"Warning: dt={dt} too large for explicit stability; using dt={dt_stable:.4e}")
        dt = dt_stable

    time = 0.0
    nsteps = int(np.ceil(t_final / dt))

    # History tracking
    time_history = []
    temp_history = []
    if save_history:
        save_every = max(1, int(save_interval / dt))
        time_history.append(0.0)
        temp_history.append(T.copy())

    for n in range(nsteps):
        T_old = T.copy()

        T = forward_euler_step(
            T, dt, k, h, rho, cp, T_inf,
            center_nodes, neighbours_dict,
            face_areas, center_distances,
            boundary_areas, volumes_dict=volumes,
            bottom_cells=bottom_cells, T_base=T_base
        )

        time += dt

        # Save history at specified intervals
        if save_history and (n + 1) % save_every == 0:
            time_history.append(time)
            temp_history.append(T.copy())

        max_diff = np.max(np.abs(T - T_old))
        if max_diff < tol:
            print(f"Converged at t = {time:.4f} s, step {n}")
            if save_history:
                time_history.append(time)
                temp_history.append(T.copy())
            break

    if save_history:
        return T, coords, center_nodes, time_history, temp_history
    else:
        return T, coords, center_nodes


# ------------------------------------------------------------
# Example of usage
# (Run the solver if the file is executed directly)
# ------------------------------------------------------------
if __name__ == "__main__":

    # Material properties of aluminum
    rho = 2700         # kg/m³
    cp  = 900          # J/kg·K
    k   = 205          # W/m·K
    h   = 300          # W/m²·K

    T_inf  = 25.0      # ambient air temperature
    T_base = 90.0      # CPU temperature

    T, coords, centers = solve_transient(
        k, h, rho, cp,
        T_inf=T_inf,
        T_base=T_base,
        dt=0.01,
        t_final=10.0,
        tol=1e-6
    )

    print("Final temperatures at center nodes:")
    for c in centers[:10]:  # print first 10
        print(c, T[c])
