# debug_neighbors.py
# Script para visualizar los vecinos de cada nodo

import numpy as np
from grid_matrices_2d_with_centers import build_fin_grid_2d
from parameters import N_HEIGHT, N_WIDTH

def print_neighbors_info():
    """
    Imprime información detallada sobre los nodos y sus vecinos
    """

    # Construir geometría con parámetros simples
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
        fin_thickness=0.002,
        fin_height=0.025,
        fin_length=0.050,
        tip_ratio=1.0,
        round_factor=0.0
    )

    print("="*80)
    print("ANÁLISIS DE NODOS Y VECINOS")
    print("="*80)
    print(f"\nGrid completo: {N_HEIGHT} rows x {N_WIDTH} cols = {N_HEIGHT * N_WIDTH} nodos totales")
    print(f"Center nodes: {len(center_nodes)} nodos\n")

    # Imprimir info de algunos nodos específicos
    print("\n" + "="*80)
    print("EJEMPLOS DE NODOS Y SUS VECINOS")
    print("="*80)
    print("\nFormato: [Center, Left, Right, Up, Down]")
    print("  - Center: nodo central")
    print("  - Left:   vecino en dirección -X (col - 2)")
    print("  - Right:  vecino en dirección +X (col + 2)")
    print("  - Up:     vecino en dirección +Y (row + 2)")
    print("  - Down:   vecino en dirección -Y (row - 2)")

    # Seleccionar algunos nodos interesantes para analizar
    examples = []

    # Nodo esquina inferior izquierda (base)
    bottom_left = 0
    if bottom_left in center_nodes:
        examples.append(("ESQUINA INFERIOR IZQUIERDA (base)", bottom_left))

    # Nodo en el centro de la base
    center_base_row = 0
    center_base_col = (N_WIDTH - 1) // 2
    center_base = center_base_row * N_WIDTH + center_base_col
    if center_base in center_nodes:
        examples.append(("CENTRO DE LA BASE", center_base))

    # Nodo en el medio del grid
    mid_row = (N_HEIGHT - 1) // 2
    mid_col = (N_WIDTH - 1) // 2
    center_mid = mid_row * N_WIDTH + mid_col
    if center_mid in center_nodes:
        examples.append(("CENTRO DEL GRID", center_mid))

    # Nodo en la punta
    top_row = N_HEIGHT - 1
    top_col = (N_WIDTH - 1) // 2
    center_top = top_row * N_WIDTH + top_col
    if center_top in center_nodes:
        examples.append(("CENTRO DE LA PUNTA", center_top))

    # Esquina superior derecha
    top_right = (N_HEIGHT - 1) * N_WIDTH + (N_WIDTH - 1)
    if top_right in center_nodes:
        examples.append(("ESQUINA SUPERIOR DERECHA (punta)", top_right))

    for label, node in examples:
        print(f"\n{'-'*80}")
        print(f"{label}: Nodo {node}")
        print(f"{'-'*80}")

        # Convertir a row, col
        row = node // N_WIDTH
        col = node % N_WIDTH

        # Coordenadas físicas
        x, y = coords[node]

        print(f"  Posición en grid: row={row}, col={col}")
        print(f"  Coordenadas físicas: X={x*1000:.4f} mm, Y={y*1000:.4f} mm")

        # Vecinos
        neighbors = neighbours_dict[node]
        print(f"\n  Vecinos: {neighbors}")
        print(f"    [0] Center: {neighbors[0]}")
        print(f"    [1] Left:   {neighbors[1]} ", end="")
        if neighbors[1] is not None:
            x_n, y_n = coords[neighbors[1]]
            print(f"-> (X={x_n*1000:.4f} mm, Y={y_n*1000:.4f} mm) DX={((x_n-x)*1000):.4f} mm")
        else:
            print("-> FRONTERA")

        print(f"    [2] Right:  {neighbors[2]} ", end="")
        if neighbors[2] is not None:
            x_n, y_n = coords[neighbors[2]]
            print(f"-> (X={x_n*1000:.4f} mm, Y={y_n*1000:.4f} mm) DX={((x_n-x)*1000):.4f} mm")
        else:
            print("-> FRONTERA")

        print(f"    [3] Up:     {neighbors[3]} ", end="")
        if neighbors[3] is not None:
            x_n, y_n = coords[neighbors[3]]
            print(f"-> (X={x_n*1000:.4f} mm, Y={y_n*1000:.4f} mm) DY={((y_n-y)*1000):.4f} mm")
        else:
            print("-> FRONTERA")

        print(f"    [4] Down:   {neighbors[4]} ", end="")
        if neighbors[4] is not None:
            x_n, y_n = coords[neighbors[4]]
            print(f"-> (X={x_n*1000:.4f} mm, Y={y_n*1000:.4f} mm) DY={((y_n-y)*1000):.4f} mm")
        else:
            print("-> FRONTERA")

    # Análisis de la base
    print("\n" + "="*80)
    print("ANÁLISIS DE LA BASE (Y mínimo)")
    print("="*80)

    y_coords = coords[center_nodes, 1]
    y_min = y_coords.min()
    y_max = y_coords.max()
    tol = 1e-12 + 1e-6 * abs(y_min)
    bottom_cells = [c for c in center_nodes if coords[c, 1] <= y_min + tol]

    print(f"\nY_min = {y_min*1000:.4f} mm")
    print(f"Y_max = {y_max*1000:.4f} mm")
    print(f"Nodos en la base: {len(bottom_cells)}")
    print(f"Nodos base: {sorted(bottom_cells)}")

    # Para cada nodo base, verificar sus vecinos
    print("\nVerificacion de vecinos de nodos base:")
    for base_node in sorted(bottom_cells):
        neighbors = neighbours_dict[base_node]
        x, y = coords[base_node]
        print(f"\n  Nodo {base_node} (X={x*1000:.4f} mm, Y={y*1000:.4f} mm):")
        print(f"    Down (hacia Y-): {neighbors[4]} {'<- DEBERIA SER FRONTERA' if neighbors[4] is not None else '<- OK: frontera'}")
        print(f"    Up (hacia Y+):   {neighbors[3]} {'<- OK: vecino interior' if neighbors[3] is not None else '<- PROBLEMA: deberia tener vecino'}")

        # Verificar que el vecino Up realmente está arriba
        if neighbors[3] is not None:
            x_up, y_up = coords[neighbors[3]]
            if y_up > y:
                print(f"      [OK] Correcto: vecino Up esta en Y={y_up*1000:.4f} mm > {y*1000:.4f} mm")
            else:
                print(f"      [ERROR] vecino Up esta en Y={y_up*1000:.4f} mm <= {y*1000:.4f} mm")

    print("\n" + "="*80)

if __name__ == "__main__":
    print_neighbors_info()
