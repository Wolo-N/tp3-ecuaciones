import numpy as np
import math

# Import your geometry builder
from grid_matrices_2d_with_centers import build_fin_grid_2d
from parameters import RHO, CP, K, H, T_INF, T_BASE, Q_REQUIRED, BASE_SIZE

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
                    dt=0.01, t_final=5.0, tol=1e-4,
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

        # Print progress every 10000 steps
        if n % 10000 == 0 and n > 0:
            print(f"  Step {n}, t = {time:.4f} s, max_diff = {max_diff:.2e}")

        if max_diff < tol:
            print(f"Converged at t = {time:.4f} s, step {n}, max_diff = {max_diff:.2e}")
            break

        time += dt

    # If we finished without converging
    if max_diff >= tol:
        print(f"Did NOT converge after {nsteps} steps (t_final = {time:.4f} s), max_diff = {max_diff:.2e}")

    return T, coords, center_nodes


def evaluate_fin_design(fin_thickness,
                        fin_height,
                        fin_length=0.05,
                        gap=0.001,  # 0.06mm gap for tighter packing
                        tip_ratio=1.0,
                        round_factor=0.5,
                        check_balance=True):
    """
    Evalúa un diseño de aleta:
    - Corre el solver transitorio para obtener T.
    - Calcula el calor disipado por UNA aleta.
    - Calcula la masa de UNA aleta.
    - Determina cuántas aletas son necesarias para disipar 500 W.
    - Determina si entran físicamente en el disipador.
    Devuelve un diccionario con toda la info del diseño.
    """

    # -------- 1. Resolver el campo de temperaturas --------
    T, coords, center_nodes = solve_transient(
        k=K, h=H, rho=RHO, cp=CP,
        T_inf=T_INF,
        T_base=T_BASE,
        dt=0.005,
        t_final=200.0,
        tol=1e-4,
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
        Q_fin += H * A_surf * (T[c] - T_INF)

    if Q_fin <= 0:
        return None

    # -------- 4. Masa de una aleta --------
    V_fin = sum(volumes_dict.values())
    m_fin = RHO * V_fin

    # -------- 5. Cuántas aletas entran físicamente --------
    pitch = fin_thickness + gap
    n_max = int(BASE_SIZE // pitch)

    # 6) Cuántas aletas necesito (real) y entero
    n_req_real = Q_REQUIRED / Q_fin
    n_req = math.ceil(n_req_real)   # número entero de aletas

    feasible = n_req <= n_max

    m_total = n_req * m_fin

    # --------------------------------------------------------
    # Heat balance verification for this fin design (opcional)
    # --------------------------------------------------------
    if check_balance:
        # Identify bottom cells again (same logic as in solve_transient)
        y_coords = coords[center_nodes, 1]
        y_min = y_coords.min()
        tol = 1e-12 + 1e-6 * abs(y_min)
        bottom_cells = [c for c in center_nodes if coords[c, 1] <= y_min + tol]

        # Run heat balance check
        check_heat_balance(T,
                            center_nodes,
                            bottom_cells,
                            neighbours_dict,
                            face_areas,
                            center_distances,
                            boundary_areas,
                            K, H, T_INF)


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

def check_heat_balance(T,
                       center_nodes,
                       bottom_cells,
                       neighbours_dict,
                       face_areas,
                       center_distances,
                       boundary_areas,
                       k, h, T_inf):
    """
    Verifica el balance de energía aislando el "Cuerpo de la Aleta".

    Balance correcto en estado estacionario:
    (Conducción desde Nodos Base hacia Nodos Cuerpo) == (Convección de Nodos Cuerpo)

    Parameters
    ----------
    T : array
        Campo de temperaturas
    center_nodes : list
        Lista de nodos centrales
    bottom_cells : list
        Lista de celdas en la base
    neighbours_dict : dict
        Diccionario de vecinos
    face_areas : dict
        Áreas de caras entre celdas
    center_distances : dict
        Distancias entre centros
    boundary_areas : dict
        Áreas de frontera para convección
    k : float
        Conductividad térmica
    h : float
        Coeficiente de convección
    T_inf : float
        Temperatura ambiente

    Returns
    -------
    tuple
        (Q_cond_in, Q_conv_body, rel_err)
    """

    # Identificar nodos del cuerpo (aquellos que NO son temperatura fija)
    base_set = set(bottom_cells)
    body_nodes = [c for c in center_nodes if c not in base_set]

    # --- 1) Calor entrando al cuerpo (Desde la Base) ---
    # Sumamos solo flujos que cruzan la frontera Base -> Cuerpo
    Q_cond_in = 0.0

    for c in bottom_cells:
        # Revisar vecinos de cada nodo base
        for j in neighbours_dict[c]:
            if j is None or j == c:
                continue

            # Si el vecino NO es base, es una conexión hacia el cuerpo de la aleta
            if j not in base_set:
                A = face_areas.get((c, j), 0.0)
                d = center_distances.get((c, j), 1.0)

                # Flux = k * A * (T_base - T_vecino) / d
                if A > 0 and d > 0:
                    Q_cond_in += k * A * (T[c] - T[j]) / d

    # --- 2) Calor saliendo del cuerpo (Convección) ---
    # Sumamos convección solo de los nodos del cuerpo
    Q_conv_body = 0.0
    for c in body_nodes:
        A_surf = boundary_areas.get(c, 0.0)
        Q_conv_body += h * A_surf * (T[c] - T_inf)

    # (Opcional) Calor perdido directamente por la base (si tiene bordes expuestos)
    Q_conv_base = 0.0
    for c in bottom_cells:
        A_surf = boundary_areas.get(c, 0.0)
        Q_conv_base += h * A_surf * (T[c] - T_inf)

    # --- 3) Error relativo sobre el cuerpo ---
    # En estado estacionario, Q_cond_in debe ser igual a Q_conv_body
    denom = max(abs(Q_cond_in), abs(Q_conv_body), 1e-12)
    rel_err = abs(Q_cond_in - Q_conv_body) / denom * 100

    Q_total_dissipated = Q_conv_body + Q_conv_base

    print("\n" + "="*40)
    print(" VERIFICACIÓN DE BALANCE TÉRMICO (CUERPO ALETA)")
    print("="*40)
    print(f"Heat Input (Cond Base->Body): {Q_cond_in:.4f} W")
    print(f"Heat Output (Conv Body):      {Q_conv_body:.4f} W")
    print("-" * 40)
    print(f"Balance Error:                {rel_err:.4f} %")
    print("-" * 40)
    print(f"Base Direct Convection:       {Q_conv_base:.4f} W")
    print(f"TOTAL Heatsink Dissipation:   {Q_total_dissipated:.4f} W")
    print("="*40 + "\n")

    return Q_cond_in, Q_conv_body, rel_err


# ------------------------------------------------------------
# Example of usage
# (Run the solver if the file is executed directly)
# ------------------------------------------------------------
if __name__ == "__main__":

    T, coords, centers = solve_transient(
        K, H, RHO, CP,
        T_inf=T_INF,
        T_base=T_BASE,
        dt=0.01,
        t_final=10.0,
        tol=1e-4,
        fin_thickness=0.0015,
        fin_height=0.020,
        fin_length=0.050,
        tip_ratio=1.0,
        round_factor=0.5
    )

    print("Final temperatures at center nodes:")
    for c in centers[:10]:  # print first 10
        print(c, T[c])


    evaluate_design = evaluate_fin_design(fin_thickness=0.0015,
                                            fin_height=0.020,
                                            fin_length=0.050,
                                            gap=0.001,
                                            tip_ratio=1.0,
                                            round_factor=0.5)
    print("Evaluation of fin design:", evaluate_design)