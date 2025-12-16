import numpy as np
from numpy import sqrt
from parameters import N_HEIGHT, N_WIDTH


# ============================================================================
# FUNCIONES DE FORMA Q9
# ============================================================================

def q9_shape_functions(xi, eta):
    """
    Calcula las funciones de forma Q9 (elemento cuadrático de 9 nodos).

    Parámetros
    ----------
    xi : float
        Coordenada natural en dirección xi ∈ [-1, 1]
    eta : float
        Coordenada natural en dirección eta ∈ [-1, 1]

    Retorna
    -------
    N : ndarray
        Vector con las 9 funciones de forma evaluadas en (xi, eta)
    """
    # Polinomios de Lagrange cuadráticos en 1D
    L1 = 0.5 * xi * (xi - 1.0)
    L2 = 1.0 - xi**2
    L3 = 0.5 * xi * (xi + 1.0)

    M1 = 0.5 * eta * (eta - 1.0)
    M2 = 1.0 - eta**2
    M3 = 0.5 * eta * (eta + 1.0)

    # Funciones de forma por producto tensorial (Q9)
    N = np.array([
        L1 * M1,  # N1 (-1,-1) esquina inferior izquierda
        L3 * M1,  # N2 ( 1,-1) esquina inferior derecha
        L3 * M3,  # N3 ( 1, 1) esquina superior derecha
        L1 * M3,  # N4 (-1, 1) esquina superior izquierda
        L2 * M1,  # N5 ( 0,-1) medio inferior
        L3 * M2,  # N6 ( 1, 0) medio derecha
        L2 * M3,  # N7 ( 0, 1) medio superior
        L1 * M2,  # N8 (-1, 0) medio izquierda
        L2 * M2   # N9 ( 0, 0) centro
    ])
    return N


# ============================================================================
# INTERPOLACIÓN CON FUNCIONES DE FORMA Q9
# ============================================================================

def q9_interpolate_points(ctrl_pts, natural_coords):
    """
    Interpola coordenadas físicas (X,Y) usando funciones de forma Q9.

    Parámetros
    ----------
    ctrl_pts : (9,2) array_like
        Coordenadas de los 9 puntos de control en el orden:
        [(-1,-1), ( 1,-1), ( 1, 1), (-1, 1),
         ( 0,-1), ( 1, 0), ( 0, 1), (-1, 0), ( 0, 0)]
    natural_coords : (m,2) array_like
        Cada fila es (xi, eta) en [-1,1]²

    Retorna
    -------
    structural_coords : (m,2) ndarray
        Puntos interpolados en coordenadas físicas
    """
    ctrl_pts = np.asarray(ctrl_pts, dtype=float).reshape(9, 2)
    natural_coords = np.asarray(natural_coords, dtype=float).reshape(-1, 2)

    # Construir matriz de funciones de forma para todos los puntos
    Nmat = np.vstack([q9_shape_functions(xi, eta)
                      for xi, eta in natural_coords])  # (m, 9)

    # Interpolar todos los puntos simultáneamente
    structural_coords = Nmat @ ctrl_pts  # (m, 2)
    return structural_coords


# ============================================================================
# SELECCIÓN DE NODOS CENTRALES (CENTROS DE VOLÚMENES DE CONTROL)
# ============================================================================

def get_center_nodes(nHeight, nWidth):
    """
    Selecciona nodos centrales tomando 1 de cada 2 nodos en ambas direcciones.

    Para una malla de 11×11, selecciona nodos en filas 0, 2, 4, 6, 8, 10 y
    columnas 0, 2, 4, 6, 8, 10, creando una submalla de 6×6 nodos centrales.

    Parámetros
    ----------
    nHeight : int
        Número de nodos en dirección eta (filas)
    nWidth : int
        Número de nodos en dirección xi (columnas)

    Retorna
    -------
    center_nodes : list
        Lista de índices globales de los nodos centrales
    center_nodes_grid : dict
        Diccionario que mapea (fila, col) → índice global de nodos centrales
    """
    center_nodes = []
    center_nodes_grid = {}

    for row in range(0, nHeight, 2):  # Filas 0, 2, 4, 6, 8, 10
        for col in range(0, nWidth, 2):  # Columnas 0, 2, 4, 6, 8, 10
            node_index = row * nWidth + col
            center_nodes.append(node_index)
            center_nodes_grid[(row, col)] = node_index

    return center_nodes, center_nodes_grid



# ============================================================================
# DEFINICIÓN DE BLOQUES (VECINDARIOS 3×3 ALREDEDOR DE CADA NODO CENTRAL)
# ============================================================================

def get_block_nodes(center_idx, nWidth, nHeight):
    """
    Obtiene los nodos del bloque 3×3 alrededor de un nodo central.

    Parámetros
    ----------
    center_idx : int
        Índice global del nodo central
    nWidth : int
        Ancho de la malla (número de nodos en dirección X)
    nHeight : int
        Altura de la malla (número de nodos en dirección Y)

    Retorna
    -------
    block : list
        Lista de índices de nodos en el bloque (hasta 9 nodos)
    """
    # Convertir índice global a (fila, columna)
    center_row = center_idx // nWidth
    center_col = center_idx % nWidth

    block = []
    # Obtener vecindario 3×3 (incluyendo el centro)
    for dr in [-1, 0, 1]:  # Desplazamiento en filas
        for dc in [-1, 0, 1]:  # Desplazamiento en columnas
            r = center_row + dr
            c = center_col + dc
            # Verificar que esté dentro de los límites de la malla
            if 0 <= r < nHeight and 0 <= c < nWidth:
                block.append(r * nWidth + c)

    return block



# ============================================================================
# CONECTIVIDAD ENTRE NODOS CENTRALES (VECINOS)
# ============================================================================

def get_center_neighbors_from_grid(center_idx, center_nodes_grid, nWidth):
    """
    Obtiene los vecinos de un nodo central en el formato:
    [Centro, Izquierda, Derecha, Arriba, Abajo]

    NOTA: Debido a cómo está configurado meshgrid:
        - Índice de fila → coordenada X (dirección del espesor de la aleta)
        - Índice de columna → coordenada Y (dirección de la altura de la aleta)

    Parámetros
    ----------
    center_idx : int
        Índice global del nodo central en la malla
    center_nodes_grid : dict
        Diccionario que mapea (fila, col) → índice global de nodos centrales
    nWidth : int
        Ancho de la malla completa

    Retorna
    -------
    neighbors : list
        [Centro, Izq, Der, Arriba, Abajo] donde None indica ausencia de vecino
    """
    # Convertir índice central a (fila, columna) en la malla global
    center_row = center_idx // nWidth
    center_col = center_idx % nWidth

    # Inicializar vecinos: [Centro, Izquierda, Derecha, Arriba, Abajo]
    neighbors = [center_idx, None, None, None, None]

    # Vecino izquierdo: fila - 2 (menor X), misma columna
    left_key = (center_row - 2, center_col)
    if left_key in center_nodes_grid:
        neighbors[1] = center_nodes_grid[left_key]

    # Vecino derecho: fila + 2 (mayor X), misma columna
    right_key = (center_row + 2, center_col)
    if right_key in center_nodes_grid:
        neighbors[2] = center_nodes_grid[right_key]

    # Vecino superior: misma fila, columna + 2 (mayor Y)
    up_key = (center_row, center_col + 2)
    if up_key in center_nodes_grid:
        neighbors[3] = center_nodes_grid[up_key]

    # Vecino inferior: misma fila, columna - 2 (menor Y)
    down_key = (center_row, center_col - 2)
    if down_key in center_nodes_grid:
        neighbors[4] = center_nodes_grid[down_key]

    return neighbors



# ============================================================================
# CÁLCULO DE ÁREAS
# ============================================================================

def heron_triangle_area(p1, p2, p3):
    """
    Calcula el área de un triángulo usando la fórmula de Herón.

    Parámetros
    ----------
    p1, p2, p3 : array_like
        Coordenadas de los tres vértices del triángulo

    Retorna
    -------
    area : float
        Área del triángulo en m²
    """
    # Calcular longitudes de los lados
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)

    # Semiperímetro
    s = (a + b + c) / 2.0

    # Fórmula de Herón: A = √[s(s-a)(s-b)(s-c)]
    val = s * (s - a) * (s - b) * (s - c)
    if val <= 0:
        return 0.0  # Triángulo degenerado o colineal; sin área
    area = sqrt(val)

    return area


def _compute_angle_from_center(node_idx, center_pos, structural_coords):
    """
    Función auxiliar para calcular el ángulo desde el centro hacia un nodo.

    Parámetros
    ----------
    node_idx : int
        Índice global del nodo
    center_pos : ndarray
        Posición del nodo central
    structural_coords : ndarray
        Coordenadas de todos los nodos

    Retorna
    -------
    angle : float
        Ángulo en radianes desde el centro hacia el nodo
    """
    node_pos = structural_coords[node_idx]
    dx = node_pos[0] - center_pos[0]
    dy = node_pos[1] - center_pos[1]
    return np.arctan2(dy, dx)


def compute_block_area(center_idx, block_nodes, structural_coords):
    """
    Calcula el área de un bloque dividiéndolo en triángulos.

    El bloque se divide conectando el nodo central con todos los nodos circundantes,
    formando triángulos con pares consecutivos de nodos circundantes.

    Parámetros
    ----------
    center_idx : int
        Índice global del nodo central
    block_nodes : list
        Lista de índices de nodos en el bloque 3×3
    structural_coords : ndarray
        Coordenadas de todos los nodos

    Retorna
    -------
    total_area : float
        Área total del bloque en m²
    """
    # Obtener posición del nodo central
    center_pos = structural_coords[center_idx]

    # Obtener nodos circundantes en orden antihorario
    # Los nodos del bloque están ordenados fila por fila, necesitamos reordenarlos
    # Disposición estándar del bloque 3×3 (fila por fila):
    # [0, 1, 2]   --> corresponde a: [SI, S, SD]  (Superior Izq, Superior, Superior Der)
    # [3, 4, 5]   --> corresponde a: [I,  C, D ]  (Izquierda, Centro, Derecha)
    # [6, 7, 8]   --> corresponde a: [II, I, ID]  (Inferior Izq, Inferior, Inferior Der)

    # Crear mapeo de fila-por-fila a orden antihorario alrededor del centro
    # Antihorario comenzando desde inferior-izquierda: II, I, ID, D, SD, S, SI, I
    if len(block_nodes) == 9:
        # Bloque completo con todos los 9 nodos
        counterclockwise_order = [6, 7, 8, 5, 2, 1, 0, 3]
        surrounding_indices = [block_nodes[i] for i in counterclockwise_order]
    else:
        # Bloque de frontera - extraer nodos circundantes (excluir centro)
        surrounding_indices = [node for node in block_nodes if node != center_idx]

        # Ordenar nodos circundantes antihorariamente según su posición
        surrounding_indices.sort(
            key=lambda node_idx: _compute_angle_from_center(
                node_idx, center_pos, structural_coords
            )
        )

    # Calcular área total sumando triángulos
    total_area = 0.0
    n = len(surrounding_indices)

    for i in range(n):
        p1 = center_pos
        p2 = structural_coords[surrounding_indices[i]]
        p3 = structural_coords[surrounding_indices[(i + 1) % n]]  # Cierra el ciclo

        triangle_area = heron_triangle_area(p1, p2, p3)
        total_area += triangle_area

    return total_area



# ============================================================================
# CONSTRUCCIÓN DE LA MALLA 2D DE LA ALETA CON DATOS GEOMÉTRICOS FVM
# ============================================================================

def build_fin_grid_2d(fin_thickness=0.002,
                      fin_height=0.025,
                      fin_length=0.050,
                      tip_ratio=1.0,
                      round_factor=0.5):
    """
    Construye la malla 2D de la aleta y todos los datos geométricos del FVM.

    Parámetros
    ----------
    fin_thickness : float
        Espesor de la base de la aleta [m]
    fin_height : float
        Altura de la aleta [m]
    fin_length : float
        Longitud de la aleta (profundidad en Z) [m]
    tip_ratio : float
        Proporción del ancho de la punta respecto a la base (1.0 = rectangular)
    round_factor : float
        Factor de redondeo de la punta (0.0 = aguda, 1.0 = redondeada)

    Retorna
    -------
    tuple
        (structural_coords, center_nodes, neighbours_dict, areas_dict,
         volumes_dict, blocks, center_distances, face_lengths, face_areas,
         boundary_areas)
    """

    # ------------------------------------------------------------------------
    # 1. DISCRETIZACIÓN EN COORDENADAS NATURALES
    # ------------------------------------------------------------------------
    nHeight = N_HEIGHT  # Número de nodos en dirección eta (debe ser impar)
    nWidth = N_WIDTH    # Número de nodos en dirección xi (debe ser impar)

    h = np.linspace(-1, 1, nHeight)  # Coordenada eta
    w = np.linspace(-1, 1, nWidth)   # Coordenada xi
    H, W = np.meshgrid(h, w)         # Malla: W→xi, H→eta

    # Matriz de coordenadas naturales (xi, eta)
    natural_coords = np.column_stack((W.ravel(), H.ravel()))

    # ------------------------------------------------------------------------
    # 2. PUNTOS DE CONTROL PARA LA GEOMETRÍA DE LA ALETA
    # ------------------------------------------------------------------------
    # La aleta es simétrica respecto a X=0 con:
    #   - Base plana en Y=0
    #   - Punta (potencialmente ahusada y redondeada) en Y=fin_height

    half_t_base = fin_thickness / 2.0
    half_t_tip = tip_ratio * half_t_base

    # Punta redondeada: las esquinas superiores bajan respecto al centro
    # round_factor = 0 → punta aguda (todas las esquinas a la misma altura)
    # round_factor = 1 → punta redondeada (esquinas bajan 10% de la altura)
    tip_corner_drop = round_factor * fin_height * 0.1
    tip_corner_height = fin_height - tip_corner_drop

    # Ancho a mitad de altura (promedio lineal entre base y punta)
    half_t_mid = 0.5 * (half_t_base + half_t_tip)

    # Definir los 9 puntos de control del elemento Q9
    ctrl = np.array([
        # Base (recta)
        [-half_t_base, 0.0],                # N1: base izquierda
        [half_t_base, 0.0],                 # N2: base derecha

        # Esquinas de la punta (bajan según round_factor para redondear)
        [half_t_tip, tip_corner_height],    # N3: esquina superior derecha
        [-half_t_tip, tip_corner_height],   # N4: esquina superior izquierda

        # Puntos intermedios
        [0.0, 0.0],                         # N5: centro base

        # Mitad de altura
        [half_t_mid, fin_height / 2],       # N6: mitad derecha
        [0.0, fin_height],                  # N7: centro punta (punto más alto)
        [-half_t_mid, fin_height / 2],      # N8: mitad izquierda
        [0.0, fin_height / 2],              # N9: centro medio
    ])




    # ------------------------------------------------------------------------
    # 3. TRANSFORMACIÓN DE COORDENADAS (NATURAL → FÍSICA)
    # ------------------------------------------------------------------------
    structural_coords = q9_interpolate_points(ctrl, natural_coords)

    # ------------------------------------------------------------------------
    # 4. IDENTIFICAR NODOS CENTRALES Y CONSTRUIR BLOQUES
    # ------------------------------------------------------------------------
    center_nodes, center_nodes_grid = get_center_nodes(nHeight, nWidth)

    # Crear diccionario que mapea nodos centrales a sus bloques
    blocks = {}
    for center in center_nodes:
        blocks[center] = get_block_nodes(center, nWidth, nHeight)

    # Construir arreglo de vecinos para todos los nodos centrales
    n_centers = len(center_nodes)
    neighbours = np.empty((n_centers, 5), dtype=object)

    for i, center in enumerate(center_nodes):
        neighbours[i, :] = get_center_neighbors_from_grid(
            center, center_nodes_grid, nWidth
        )

    # Crear diccionario para búsqueda rápida: índice_nodo → lista_vecinos
    center_node_to_idx = {node: i for i, node in enumerate(center_nodes)}
    neighbours_dict = {node: neighbours[i] for i, node in enumerate(center_nodes)}

    # ------------------------------------------------------------------------
    # 5. CALCULAR ÁREAS 2D DE LOS VOLÚMENES DE CONTROL
    # ------------------------------------------------------------------------
    areas = np.zeros(n_centers, dtype=float)
    areas_dict = {}

    for i, center in enumerate(center_nodes):
        block = blocks[center]
        areas[i] = compute_block_area(center, block, structural_coords)
        areas_dict[center] = areas[i]

    # ------------------------------------------------------------------------
    # 6. CALCULAR VOLÚMENES 3D
    # ------------------------------------------------------------------------
    # Malla 2D = (espesor × altura); extruir en dirección Z por fin_length
    # Volumen 3D = Área 2D × Longitud de extrusión

    volumes = areas * fin_length
    volumes_dict = {center_nodes[i]: volumes[i] for i in range(len(center_nodes))}

    # ------------------------------------------------------------------------
    # 7. DISTANCIAS CENTRO A CENTRO d_ij
    # ------------------------------------------------------------------------
    center_distances = {}  # Clave: (i, j) usando IDs globales de nodos centrales

    for c_idx, c in enumerate(center_nodes):
        cx, cy = structural_coords[c]
        for neigh in neighbours_dict[c]:
            if neigh is None or neigh == c:
                continue
            nx, ny = structural_coords[neigh]
            d = np.linalg.norm([nx - cx, ny - cy])
            center_distances[(c, neigh)] = d

    # ------------------------------------------------------------------------
    # 8. LONGITUDES Y ÁREAS DE CARAS
    # ------------------------------------------------------------------------
    # Cada volumen de control tiene un bloque (lista de nodos circundantes)
    # Encontramos aristas compartidas entre bloques

    face_lengths = {}  # Longitud 2D de la cara entre VCs [m]
    face_areas = {}    # Área 3D de la cara = face_length × fin_length [m²]

    for c_idx, c in enumerate(center_nodes):
        block_c = blocks[c]  # Nodos que rodean el VC c

        for neigh in neighbours_dict[c]:
            if neigh is None or neigh == c:
                continue

            block_n = blocks[neigh]

            # Encontrar nodos compartidos entre los polígonos de bloques
            shared = list(set(block_c).intersection(set(block_n)))
            L = 0.0

            if len(shared) >= 2:
                # Elegir la arista más larga entre nodos compartidos
                # (maneja limpiamente superposiciones de 3 nodos)
                max_len = 0.0
                for i in range(len(shared)):
                    for j in range(i + 1, len(shared)):
                        p1 = structural_coords[shared[i]]
                        p2 = structural_coords[shared[j]]
                        candidate = np.linalg.norm(p2 - p1)
                        if candidate > max_len:
                            max_len = candidate
                L = max_len

            face_lengths[(c, neigh)] = L
            face_areas[(c, neigh)] = L * fin_length  # Área 3D de la cara

    # ------------------------------------------------------------------------
    # 9. ÁREAS SUPERFICIALES DE FRONTERA (PARA CONVECCIÓN)
    # ------------------------------------------------------------------------
    boundary_areas = {}  # Área expuesta para cada VC de frontera

    for c_idx, c in enumerate(center_nodes):
        A_surf = 0.0  # Acumular área superficial

        # Agregar áreas de caras FRONTAL y POSTERIOR (en dirección Z)
        # Estas caras se crean al extruir la geometría 2D por fin_length
        # Tanto frontal (Z=0) como posterior (Z=fin_length) están expuestas al aire
        A_surf += 2 * areas_dict[c]  # Áreas frontal + posterior

        for direction_idx, neigh in enumerate(neighbours_dict[c]):
            if neigh is None:
                # Cara de frontera: encontrar la longitud de cara correspondiente
                block_c = blocks[c]

                # Determinar arista de frontera seleccionando nodos en la coordenada
                # más externa
                # direction_idx: 0=Centro, 1=Izq, 2=Der, 3=Arriba, 4=Abajo
                # Izq/Der (1,2): fronteras en dirección X (eje 0)
                # Arriba/Abajo (3,4): fronteras en dirección Y (eje 1)
                axis = 0 if direction_idx in (1, 2) else 1
                coords_block = structural_coords[block_c][:, axis]

                # Izquierda (1) o Abajo (4) → arista mínima
                # Derecha (2) o Arriba (3) → arista máxima
                if direction_idx in (1, 4):
                    edge_coord = np.min(coords_block)
                else:
                    edge_coord = np.max(coords_block)

                tol = 1e-12 + 1e-6 * abs(edge_coord)
                edge_nodes = [
                    n for n in block_c
                    if abs(structural_coords[n][axis] - edge_coord) <= tol
                ]

                # Elegir el segmento más largo entre nodos de arista
                # (cubre >2 nodos colineales)
                edge_len = 0.0
                if len(edge_nodes) >= 2:
                    for i in range(len(edge_nodes)):
                        for j in range(i + 1, len(edge_nodes)):
                            p1 = structural_coords[edge_nodes[i]]
                            p2 = structural_coords[edge_nodes[j]]
                            edge_len = max(edge_len, np.linalg.norm(p2 - p1))

                # Convertir a área 3D: arista × fin_length
                A_surf += edge_len * fin_length

        boundary_areas[c] = A_surf

    # ------------------------------------------------------------------------
    # 10. RETORNAR TODOS LOS DATOS GEOMÉTRICOS
    # ------------------------------------------------------------------------
    return (structural_coords,
            center_nodes,
            neighbours_dict,
            areas_dict,
            volumes_dict,
            blocks,
            center_distances,
            face_lengths,
            face_areas,
            boundary_areas)
