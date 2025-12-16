import numpy as np
import math
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Importar constructor de geometría y parámetros
from finGridwCenters import build_fin_grid_2d
from parameters import RHO, CP, K, H, T_INF, T_BASE, Q_REQUIRED, BASE_SIZE, GAP

# ============================================================================
# FUNCIONES DE ESTABILIDAD Y ACTUALIZACIÓN
# ============================================================================

def stable_timestep(center_nodes, neighbours_dict, face_areas, center_distances,
                    boundary_areas, volumes_dict, rho, cp, k, h, safety=0.5):
    """
    Estima un timestep estable para el método explícito usando criterio de capacitancia global:
        dt_max = safety * (rho*cp*V) / (sum(k*A/d) + h*A_surf)

    Devuelve el mínimo dt_max sobre todos los volúmenes de control.

    Parameters
    ----------
    center_nodes : list
        Lista de índices de nodos centrales
    neighbours_dict : dict
        Diccionario de vecinos para cada nodo
    face_areas : dict
        Áreas de caras entre celdas
    center_distances : dict
        Distancias entre centros de celdas
    boundary_areas : dict
        Áreas de frontera para convección
    volumes_dict : dict
        Volúmenes de control
    rho : float
        Densidad [kg/m³]
    cp : float
        Capacidad calorífica [J/kg·K]
    k : float
        Conductividad térmica [W/m·K]
    h : float
        Coeficiente de convección [W/m²·K]
    safety : float
        Factor de seguridad (default: 0.5)

    Returns
    -------
    float
        Timestep estable mínimo
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


def update_one_cell(c, T, dt, k, h, rho, cp, T_inf,
                    neighbours_dict, face_areas, center_distances,
                    boundary_areas, volumes_dict):
    """
    Actualización Forward Euler para un volumen de control individual.

    Calcula el cambio de temperatura en un CV debido a:
    - Conducción con vecinos: Q_cond = Σ k*A*(T_j - T_i)/d
    - Convección con ambiente: Q_conv = -h*A_surf*(T_i - T_inf)

    Parameters
    ----------
    c : int
        Índice del volumen de control
    T : ndarray
        Campo de temperaturas actual
    dt : float
        Paso de tiempo
    k : float
        Conductividad térmica [W/m·K]
    h : float
        Coeficiente de convección [W/m²·K]
    rho : float
        Densidad [kg/m³]
    cp : float
        Capacidad calorífica [J/kg·K]
    T_inf : float
        Temperatura ambiente [°C]
    neighbours_dict : dict
        Diccionario de vecinos
    face_areas : dict
        Áreas de caras entre celdas
    center_distances : dict
        Distancias entre centros
    boundary_areas : dict
        Áreas de frontera
    volumes_dict : dict
        Volúmenes de control

    Returns
    -------
    float
        Nueva temperatura del CV
    """
    Q_cond = 0.0

    # Término de conducción
    for j in neighbours_dict[c]:
        if j is None or j == c:
            continue
        A = face_areas[(c, j)]
        d = center_distances[(c, j)]
        Q_cond += k * A * (T[j] - T[c]) / d

    # Término de convección
    A_surf = boundary_areas[c]
    Q_conv = -h * A_surf * (T[c] - T_inf)

    # Actualización explícita: T_new = T_old + dt * Q_total / (rho * cp * V)
    V = volumes_dict[c]
    T_new = T[c] + dt * (Q_cond + Q_conv) / (rho * cp * V)

    return T_new


def forward_euler_step(T, dt, k, h, rho, cp, T_inf,
                       center_nodes, neighbours_dict,
                       face_areas, center_distances,
                       boundary_areas, volumes_dict,
                       bottom_cells, T_base):
    """
    Un paso completo de Forward Euler para todos los volúmenes de control.

    Actualiza el campo de temperaturas en todo el dominio y aplica
    condiciones de borde de Dirichlet en la base.

    Parameters
    ----------
    T : ndarray
        Campo de temperaturas actual
    dt : float
        Paso de tiempo
    k, h, rho, cp : float
        Propiedades térmicas del material
    T_inf : float
        Temperatura ambiente [°C]
    center_nodes : list
        Lista de nodos centrales
    neighbours_dict : dict
        Diccionario de vecinos
    face_areas : dict
        Áreas de caras
    center_distances : dict
        Distancias entre centros
    boundary_areas : dict
        Áreas de frontera
    volumes_dict : dict
        Volúmenes de control
    bottom_cells : list
        Lista de celdas en la base (BC Dirichlet)
    T_base : float
        Temperatura de la base [°C]

    Returns
    -------
    ndarray
        Campo de temperaturas actualizado
    """
    T_new = T.copy()

    # Actualizar solo nodos centrales
    for c in center_nodes:
        T_new[c] = update_one_cell(c, T, dt, k, h, rho, cp, T_inf,
                                   neighbours_dict, face_areas,
                                   center_distances, boundary_areas,
                                   volumes_dict)

    # Imponer condición de borde Dirichlet en la base del CPU
    for c in bottom_cells:
        T_new[c] = T_base

    return T_new


# ============================================================================
# SOLVER TRANSITORIO (Forward Euler explícito)
# ============================================================================

def solve_transient(k, h, rho, cp, T_inf, T_base,
                    dt=0.01, t_final=5.0, tol=1e-4,
                    fin_thickness=0.0015, fin_height=0.020,
                    fin_length=0.050, tip_ratio=1.0, round_factor=0.5):
    """
    Resuelve el campo de temperaturas transitorio usando Forward Euler explícito.

    Integra las ecuaciones de conducción-convección en el tiempo hasta alcanzar
    estado estacionario o el tiempo final especificado.

    NOTA: Este solver es LENTO. Para optimización, usar solve_steady_state().

    Parameters
    ----------
    k : float
        Conductividad térmica [W/m·K]
    h : float
        Coeficiente de convección [W/m²·K]
    rho : float
        Densidad [kg/m³]
    cp : float
        Capacidad calorífica [J/kg·K]
    T_inf : float
        Temperatura ambiente [°C]
    T_base : float
        Temperatura de la base [°C]
    dt : float
        Paso de tiempo inicial [s] (se ajusta automáticamente para estabilidad)
    t_final : float
        Tiempo final de simulación [s]
    tol : float
        Tolerancia de convergencia para estado estacionario
    fin_thickness, fin_height, fin_length, tip_ratio, round_factor : float
        Parámetros geométricos de la aleta

    Returns
    -------
    T : ndarray
        Campo de temperaturas [°C]
    coords : ndarray
        Coordenadas de nodos
    center_nodes : list
        Índices de nodos centrales
    """

    # Cargar geometría
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

    # Crear array de temperaturas para TODOS los nodos
    T = np.ones(len(coords)) * T_inf

    # Identificar CVs de la base (coordenada y mínima entre nodos centrales)
    y_coords = coords[center_nodes, 1]
    y_min = y_coords.min()
    tol_bc = 1e-12 + 1e-6 * abs(y_min)
    bottom_cells = [c for c in center_nodes if coords[c, 1] <= y_min + tol_bc]

    # Aplicar temperatura de base inicialmente
    for c in bottom_cells:
        T[c] = T_base

    # Verificar estabilidad y ajustar dt si es necesario
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

        # Imprimir progreso cada 10000 pasos
        if n % 10000 == 0 and n > 0:
            print(f"  Step {n}, t = {time:.4f} s, max_diff = {max_diff:.2e}")

        if max_diff < tol:
            print(f"Converged at t = {time:.4f} s, step {n}, max_diff = {max_diff:.2e}")
            break

        time += dt

    # Si terminamos sin converger
    if max_diff >= tol:
        print(f"Did NOT converge after {nsteps} steps (t_final = {time:.4f} s), max_diff = {max_diff:.2e}")

    return T, coords, center_nodes


# ============================================================================
# SOLVER ESTACIONARIO (Sistema lineal directo)
# ============================================================================

def solve_steady_state(k, h, T_inf, T_base,
                       fin_thickness=0.0015, fin_height=0.020,
                       fin_length=0.050, tip_ratio=1.0, round_factor=0.5):
    """
    Resuelve el campo de temperaturas en estado estacionario usando un sistema lineal.

    Mucho más rápido que el solver transitorio - RECOMENDADO PARA OPTIMIZACIÓN.

    En estado estacionario: 0 = Q_cond + Q_conv
    Para cada nodo i:
        Σ_j k*A_ij*(T_j - T_i)/d_ij - h*A_surf_i*(T_i - T_inf) = 0

    Esto forma un sistema lineal: [A]{T} = {b}

    Parameters
    ----------
    k : float
        Conductividad térmica [W/m·K]
    h : float
        Coeficiente de convección [W/m²·K]
    T_inf : float
        Temperatura ambiente [°C]
    T_base : float
        Temperatura de la base [°C]
    fin_thickness, fin_height, fin_length, tip_ratio, round_factor : float
        Parámetros geométricos de la aleta

    Returns
    -------
    T : ndarray
        Campo de temperaturas [°C]
    coords : ndarray
        Coordenadas de nodos
    center_nodes : list
        Índices de nodos centrales
    """

    # Cargar geometría
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

    # Identificar CVs de la base (BC Dirichlet)
    y_coords = coords[center_nodes, 1]
    y_min = y_coords.min()
    tol_bc = 1e-12 + 1e-6 * abs(y_min)
    bottom_cells = set([c for c in center_nodes if coords[c, 1] <= y_min + tol_bc])

    # Crear mapeo: índice global de nodo -> índice de fila en la matriz
    # Solo center_nodes participan en el sistema
    node_to_row = {c: i for i, c in enumerate(center_nodes)}
    n_equations = len(center_nodes)

    # Construir sistema lineal disperso A*T_centers = b
    # Usar lil_matrix para construcción eficiente
    A = lil_matrix((n_equations, n_equations))
    b = np.zeros(n_equations)

    for c in center_nodes:
        i = node_to_row[c]  # Índice de fila para nodo c

        if c in bottom_cells:
            # BC Dirichlet: T[c] = T_base
            A[i, i] = 1.0
            b[i] = T_base
        else:
            # Balance de energía: Σ_j k*A_ij*(T_j - T_i)/d_ij - h*A_surf_i*(T_i - T_inf) = 0
            # Reordenado: Σ_j (k*A_ij/d_ij)*T_j - T_i*Σ_j(k*A_ij/d_ij + h*A_surf_i) = -h*A_surf_i*T_inf

            diag_sum = 0.0

            # Términos de conducción (fuera de diagonal y contribución diagonal)
            for j in neighbours_dict[c]:
                if j is None or j == c:
                    continue

                A_face = face_areas[(c, j)]
                d = center_distances[(c, j)]

                coeff = k * A_face / d

                # Fuera de diagonal: coeficiente para T_j
                j_row = node_to_row[j]  # Índice de fila para vecino j
                A[i, j_row] = coeff

                # Contribución diagonal desde conducción
                diag_sum += coeff

            # Término de convección (contribución diagonal)
            A_surf = boundary_areas[c]
            diag_sum += h * A_surf

            # Entrada diagonal
            A[i, i] = -diag_sum

            # RHS: término fuente desde convección
            b[i] = -h * A_surf * T_inf

    # Convertir a formato CSR para solución eficiente
    A_csr = A.tocsr()

    # Resolver sistema lineal para temperaturas de center_nodes
    T_centers = spsolve(A_csr, b)

    # Construir array completo de temperaturas (incluyendo todos los nodos)
    n_nodes = len(coords)
    T = np.ones(n_nodes) * T_inf  # Inicializar todos los nodos a T_inf

    # Llenar temperaturas resueltas para nodos centrales
    for c, i in node_to_row.items():
        T[c] = T_centers[i]

    return T, coords, center_nodes


# ============================================================================
# EVALUACIÓN DE DISEÑO COMPLETO
# ============================================================================

def evaluate_fin_design(fin_thickness,
                        fin_height,
                        fin_length=0.05,
                        gap=GAP,  # Espacio entre aletas desde parameters.py
                        tip_ratio=1.0,
                        round_factor=0.5,
                        check_balance=True,
                        use_steady_state=True):
    """
    Evalúa un diseño completo de aleta y calcula métricas de desempeño.

    Proceso:
    1. Resuelve el campo de temperaturas (estacionario o transitorio)
    2. Calcula el calor disipado por UNA aleta
    3. Calcula la masa de UNA aleta
    4. Determina cuántas aletas son necesarias para disipar Q_REQUIRED
    5. Verifica si caben físicamente en el disipador
    6. (Opcional) Verifica balance térmico

    Parameters
    ----------
    fin_thickness : float
        Espesor de la aleta [m]
    fin_height : float
        Altura de la aleta [m]
    fin_length : float
        Longitud de la aleta [m]
    gap : float
        Espacio entre aletas consecutivas [m] (default: GAP desde parameters.py)
    tip_ratio : float
        Relación de ahusamiento (1.0 = rectangular, <1.0 = ahusado)
    round_factor : float
        Factor de redondeo de esquinas (0.0 = esquinas agudas, 1.0 = redondeado completo)
    check_balance : bool
        Si True, verifica balance térmico e imprime resultados
    use_steady_state : bool
        Si True, usa solver estacionario (RÁPIDO - recomendado para optimización)
        Si False, usa solver transitorio Forward Euler (LENTO - solo para verificación)

    Returns
    -------
    dict or None
        Diccionario con las siguientes claves:
        - thickness : float - Espesor de la aleta [m]
        - height : float - Altura de la aleta [m]
        - tip_ratio : float - Relación de ahusamiento
        - round_factor : float - Factor de redondeo
        - m_fin : float - Masa de una aleta [kg]
        - Q_fin : float - Calor disipado por una aleta [W]
        - n_required : int - Número de aletas necesarias
        - n_max : int - Número máximo de aletas que caben
        - feasible : bool - Si el diseño es factible físicamente
        - m_total : float - Masa total del disipador [kg]

        Devuelve None si Q_fin <= 0 (diseño inválido)
    """

    # -------- 1. Resolver el campo de temperaturas --------
    if use_steady_state:
        # Solver estacionario directo (RÁPIDO - recomendado para optimización)
        T, coords, center_nodes = solve_steady_state(
            k=K, h=H,
            T_inf=T_INF,
            T_base=T_BASE,
            fin_thickness=fin_thickness,
            fin_height=fin_height,
            fin_length=fin_length,
            tip_ratio=tip_ratio,
            round_factor=round_factor
        )
    else:
        # Solver transitorio Forward Euler (LENTO - solo para verificación)
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

    # -------- 6. Cuántas aletas necesito (real y entero) --------
    n_req_real = Q_REQUIRED / Q_fin
    n_req = math.ceil(n_req_real)   # número entero de aletas

    feasible = n_req <= n_max

    m_total = n_req * m_fin

    # -------- 7. Verificación de balance térmico (opcional) --------
    if check_balance:
        # Identificar celdas de la base (misma lógica que en solve_transient)
        y_coords = coords[center_nodes, 1]
        y_min = y_coords.min()
        tol = 1e-12 + 1e-6 * abs(y_min)
        bottom_cells = [c for c in center_nodes if coords[c, 1] <= y_min + tol]

        # Ejecutar verificación de balance térmico
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

# ============================================================================
# VERIFICACIÓN DE BALANCE TÉRMICO
# ============================================================================

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

    Método:
    1. Separa el dominio en "Base" (nodos con T fija) y "Cuerpo" (nodos libres)
    2. Calcula flujo conductivo que entra al cuerpo desde la base
    3. Calcula flujo convectivo que sale del cuerpo al ambiente
    4. Compara ambos flujos (deberían ser iguales en estado estacionario)

    Parameters
    ----------
    T : ndarray
        Campo de temperaturas [°C]
    center_nodes : list
        Lista de nodos centrales
    bottom_cells : list
        Lista de celdas en la base (BC Dirichlet)
    neighbours_dict : dict
        Diccionario de vecinos
    face_areas : dict
        Áreas de caras entre celdas [m²]
    center_distances : dict
        Distancias entre centros [m]
    boundary_areas : dict
        Áreas de frontera para convección [m²]
    k : float
        Conductividad térmica [W/m·K]
    h : float
        Coeficiente de convección [W/m²·K]
    T_inf : float
        Temperatura ambiente [°C]

    Returns
    -------
    tuple
        (Q_cond_in, Q_conv_body, rel_err)
        - Q_cond_in : float - Calor entrando al cuerpo por conducción [W]
        - Q_conv_body : float - Calor saliendo del cuerpo por convección [W]
        - rel_err : float - Error relativo del balance [%]
    """

    # Identificar nodos del cuerpo (aquellos que NO son temperatura fija)
    base_set = set(bottom_cells)
    body_nodes = [c for c in center_nodes if c not in base_set]

    # --- 1. Calor entrando al cuerpo (Desde la Base) ---
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

                # Flujo = k * A * (T_base - T_vecino) / d
                if A > 0 and d > 0:
                    Q_cond_in += k * A * (T[c] - T[j]) / d

    # --- 2. Calor saliendo del cuerpo (Convección) ---
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

    # --- 3. Error relativo sobre el cuerpo ---
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


# ============================================================================
# EJEMPLO DE USO
# (Ejecuta el solver si el archivo se corre directamente)
# ============================================================================
if __name__ == "__main__":

    print("="*60)
    print("COMPARACIÓN: SOLVER TRANSIENTE vs ESTACIONARIO")
    print("="*60)

    geom_kwargs = {
        "fin_thickness": 0.0015,
        "fin_height": 0.020,
        "fin_length": 0.050,
        "tip_ratio": 1.0,
        "round_factor": 0.5,
    }

    transient_kwargs = {
        "dt": 0.01,
        "t_final": 10.0,
        "tol": 1e-4,
    }

    print(">> Ejecutando solver transitorio Forward Euler...")
    T_trans, coords_trans, centers_trans = solve_transient(
        K, H, RHO, CP,
        T_inf=T_INF,
        T_base=T_BASE,
        dt=transient_kwargs["dt"],
        t_final=transient_kwargs["t_final"],
        tol=transient_kwargs["tol"],
        **geom_kwargs
    )

    print(">> Ejecutando solver estacionario (sistema lineal)...")
    T_steady, coords_steady, centers_steady = solve_steady_state(
        K, H,
        T_inf=T_INF,
        T_base=T_BASE,
        **geom_kwargs
    )

    centers_trans_set = set(centers_trans)
    centers_steady_set = set(centers_steady)
    if centers_trans_set != centers_steady_set:
        print("Warning: conjuntos de nodos centrales no coinciden exactamente; se compararán nodos comunes.")

    common_centers = sorted(list(centers_trans_set & centers_steady_set))
    diff = np.array([T_trans[c] - T_steady[c] for c in common_centers])
    abs_diff = np.abs(diff)
    max_idx = int(np.argmax(abs_diff))
    worst_node = common_centers[max_idx]

    print("\nResultados de comparación (nodos centrales comunes):")
    print(f"  Nodos comparados:         {len(common_centers)}")
    print(f"  Máx |ΔT|:                 {abs_diff[max_idx]:.4e} °C (nodo {worst_node})")
    print(f"  Promedio |ΔT|:            {np.mean(abs_diff):.4e} °C")
    print(f"  RMS ΔT:                   {math.sqrt(np.mean(diff**2)):.4e} °C")

    print("\nTemperaturas en primeros 5 nodos centrales comunes:")
    for c in common_centers[:5]:
        print(f"  Nodo {c:4d} -> T_trans = {T_trans[c]:.2f} °C | T_steady = {T_steady[c]:.2f} °C")

    print("\n" + "="*60)
    print("EJEMPLO: Evaluación de Diseño")
    print("="*60)

    evaluate_design = evaluate_fin_design(
        fin_thickness=geom_kwargs["fin_thickness"],
        fin_height=geom_kwargs["fin_height"],
        fin_length=geom_kwargs["fin_length"],
        tip_ratio=geom_kwargs["tip_ratio"],
        round_factor=geom_kwargs["round_factor"]
    )

    print("\nResultados de evaluación:")
    print(f"  Espesor: {evaluate_design['thickness']*1000:.3f} mm")
    print(f"  Altura: {evaluate_design['height']*1000:.1f} mm")
    print(f"  Masa por aleta: {evaluate_design['m_fin']*1000:.2f} g")
    print(f"  Calor por aleta: {evaluate_design['Q_fin']:.2f} W")
    print(f"  Aletas necesarias: {evaluate_design['n_required']}")
    print(f"  Aletas que caben: {evaluate_design['n_max']}")
    print(f"  Factible: {evaluate_design['feasible']}")
    print(f"  Masa total: {evaluate_design['m_total']:.3f} kg")
