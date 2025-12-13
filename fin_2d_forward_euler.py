import numpy as np

# Import your geometry builder
from grid_matrices_2d_with_centers import build_fin_grid_2d

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
                    fin_thickness=0.0015, fin_height=0.020, 
                    fin_length=0.050, tip_ratio=1.0, round_factor=0.5):

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
     boundary_areas) = build_fin_grid_2d(fin_thickness=fin_thickness,
                                            fin_height=fin_height,
                                            fin_length=fin_length,
                                            tip_ratio=tip_ratio,
                                            round_factor=round_factor)

    # Create temperature array for ALL nodes
    T = np.ones(len(coords)) * T_inf

    # Identify bottom boundary CVs via geometry (lowest y among center nodes)
    y_coords = coords[center_nodes, 1]
    y_min = y_coords.min()
    tol = 1e-12 + 1e-6 * abs(y_min)
    bottom_cells = [c for c in center_nodes if coords[c, 1] <= y_min + tol]

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

    for n in range(nsteps):
        T_old = T.copy()

        T = forward_euler_step(
            T, dt, k, h, rho, cp, T_inf,
            center_nodes, neighbours_dict,
            face_areas, center_distances,
            boundary_areas, volumes_dict=volumes,
            bottom_cells=bottom_cells, T_base=T_base
        )

        max_diff = np.max(np.abs(T - T_old))
        if max_diff < tol:
            print(f"Converged at t = {time:.4f} s, step {n}")
            break

        time += dt

    return T, coords, center_nodes

def evaluate_fin_design(fin_thickness, 
                        fin_height, 
                        fin_length=0.05, 
                        gap=0.001, 
                        tip_ratio=1.0, 
                        round_factor=0.5):
    """
    Evalúa un diseño de aleta:
    - Corre el solver transitorio para obtener T.
    - Calcula el calor disipado por UNA aleta.
    - Calcula la masa de UNA aleta.
    - Determina cuántas aletas son necesarias para disipar 500 W.
    - Determina si entran físicamente en el disipador.
    Devuelve un diccionario con toda la info del diseño.
    """

    # --- Parámetros físicos globales del problema ---
    rho = 2700       # aluminio
    cp  = 900
    k   = 205
    h   = 300        # W/m2K
    Q_REQUIRED = 500 # W requeridos
    T_INF = 45       # peor caso ambiente
    T_BASE = 90
    BASE_SIZE = 0.05 # 50 mm

    # -------- 1. Resolver el campo de temperaturas --------
    T, coords, center_nodes = solve_transient(
        k=k, h=h, rho=rho, cp=cp,
        T_inf=T_INF,
        T_base=T_BASE,
        dt=0.005,
        t_final=5.0,
        tol=1e-6,
        fin_thickness=fin_thickness, 
        fin_height=fin_height, 
        fin_length=fin_length, 
        tip_ratio=tip_ratio,
        round_factor=round_factor
    )

    # -------- 2. Recuperar la geometría completa (incluye boundary_areas) --------
    (coords,
     center_nodes,
     neighbours_dict,
     areas_dict,
     volumes_dict,
     blocks,
     center_distances,
     face_lengths,
     face_areas,
     boundary_areas) = build_fin_grid_2d(
         fin_thickness=fin_thickness,
         fin_height=fin_height,
         fin_length=fin_length,
         tip_ratio=tip_ratio, 
         round_factor=round_factor
    )

    # -------- 3. Calor disipado por UNA aleta --------
    Q_fin = 0.0
    for c in center_nodes:
        A_surf = boundary_areas[c]
        Q_fin += h * A_surf * (T[c] - T_INF)

    if Q_fin <= 0:
        return None

    # -------- 4. Masa de una aleta --------
    V_fin = sum(volumes_dict.values())
    m_fin = rho * V_fin

    # -------- 5. Cuántas aletas entran físicamente --------
    pitch = fin_thickness + gap
    n_per_row = int(BASE_SIZE // pitch)
    n_max = n_per_row ** 2  # arreglo cuadrado de alet**_

    # -------- 6. Cuántas aletas se necesitan para disipar 500 W --------
    n_req = Q_REQUIRED / Q_fin
    feasible = n_req <= n_max

    m_total = n_req * m_fin

    return {
        "thickness": fin_thickness,
        "height": fin_height,
        "tip_ratio": tip_ratio,
        "round_factor": round_factor,
        "m_fin": m_fin,
        "Q_fin": Q_fin,
        "n_required": n_req,
        "n_max": n_max,
        "feasible": feasible,
        "m_total": m_total
    }

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
        tol=1e-6,
        fin_thickness=0.0015,
        fin_height=0.020,
        fin_length=0.050,
        tip_ratio=1.0,
        round_factor=0.5
    )

    print("Final temperatures at center nodes:")
    for c in centers[:10]:  # print first 10
        print(c, T[c])


    evaluate_design = evaluate_fin_design(fin_thickness=0.0015, fin_height=0.020)
    print("Evaluation of fin design:", evaluate_design)
