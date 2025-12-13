# verify_heat_balance.py
# Verificación del balance de energía en la aleta

import numpy as np
from fin_2d_forward_euler import solve_transient
from grid_matrices_2d_with_centers import build_fin_grid_2d
from parameters import RHO, CP, K, H, T_INF, T_BASE

def verify_heat_balance(fin_thickness=0.001, fin_height=0.025, fin_length=0.05,
                        tip_ratio=1.0, round_factor=0.0):
    """
    Verifica el balance de energía en la aleta:
    - Calor entrante por la base (conducción)
    - Calor saliente por convección (todas las superficies)

    Para estado estacionario debe cumplirse: Q_in ≈ Q_out

    Parameters
    ----------
    fin_thickness : float
        Espesor de la aleta [m]
    fin_height : float
        Altura de la aleta [m]
    fin_length : float
        Longitud de la aleta en dirección Z [m]
    tip_ratio : float
        Relación entre ancho de punta y base
    round_factor : float
        Factor de redondez de la punta

    Returns
    -------
    dict
        Diccionario con Q_in, Q_out, error relativo, y detalles
    """

    print("="*70)
    print("VERIFICACIÓN DEL BALANCE DE ENERGÍA")
    print("="*70)

    # Resolver campo de temperaturas
    print("\n1. Resolviendo campo de temperaturas...")
    T, coords, center_nodes = solve_transient(
        k=K, h=H, rho=RHO, cp=CP,
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

    # Construir geometría completa
    print("\n2. Construyendo geometría FVM...")
    (structural_coords, center_nodes_list, neighbours_dict, areas_dict,
     volumes_dict, blocks, center_distances, face_lengths,
     face_areas, boundary_areas) = build_fin_grid_2d(
        fin_thickness=fin_thickness,
        fin_height=fin_height,
        fin_length=fin_length,
        tip_ratio=tip_ratio,
        round_factor=round_factor
    )

    # Identificar nodos de la base (y mínima)
    y_coords = structural_coords[center_nodes_list, 1]
    y_min = y_coords.min()
    tol = 1e-12 + 1e-6 * abs(y_min)
    bottom_cells = [c for c in center_nodes_list if structural_coords[c, 1] <= y_min + tol]

    print(f"\n3. Analizando balance térmico...")
    print(f"   - Nodos en la base: {len(bottom_cells)}")
    print(f"   - Nodos totales: {len(center_nodes_list)}")

    # ========================================================================
    # CALOR ENTRANTE POR LA BASE (conducción desde T_base hacia el interior)
    # Siguiendo la referencia del código 1D (fin_fvm_1d.py líneas 108-120):
    # Q_in = conducción hacia vecinos + convección desde superficie de nodos base
    # ========================================================================
    Q_in_total = 0.0
    Q_in_conduction = 0.0
    Q_in_convection_base = 0.0
    Q_in_details = []

    for base_node in bottom_cells:
        Q_in_node_cond = 0.0
        Q_in_node_conv = 0.0

        # 1) Conducción: Buscar vecinos hacia arriba (interior de la aleta)
        for neighbor in neighbours_dict[base_node]:
            if neighbor is None or neighbor == base_node:
                continue

            # Solo contar flujo hacia arriba (interior)
            if structural_coords[neighbor, 1] > structural_coords[base_node, 1] + tol:
                A_face = face_areas[(base_node, neighbor)]
                d = center_distances[(base_node, neighbor)]

                # Flujo de calor de base (T_BASE) hacia vecino (T[neighbor])
                # Q = k * A * (T_hot - T_cold) / d
                Q_flux = K * A_face * (T_BASE - T[neighbor]) / d
                Q_in_node_cond += Q_flux

        # 2) Convección desde las superficies expuestas del nodo base
        # (frontal, trasera, y laterales - NO incluye la cara y=0 que está en contacto con la fuente)
        A_surf_base = boundary_areas[base_node]
        Q_in_node_conv = H * A_surf_base * (T_BASE - T_INF)

        Q_in_conduction += Q_in_node_cond
        Q_in_convection_base += Q_in_node_conv
        Q_in_total += Q_in_node_cond + Q_in_node_conv
        Q_in_details.append((base_node, Q_in_node_cond, Q_in_node_conv))

    # ========================================================================
    # CALOR SALIENTE POR CONVECCIÓN (todas las superficies)
    # ========================================================================
    Q_out_total = 0.0
    Q_out_details = []

    for node in center_nodes_list:
        A_surf = boundary_areas[node]
        # Q_conv = h * A * (T_surface - T_ambient)
        Q_conv = H * A_surf * (T[node] - T_INF)
        Q_out_total += Q_conv
        Q_out_details.append((node, Q_conv, A_surf))

    # ========================================================================
    # ANÁLISIS DEL BALANCE
    # ========================================================================
    error_abs = abs(Q_in_total - Q_out_total)
    error_rel = (error_abs / Q_in_total) * 100 if Q_in_total > 0 else 0.0

    print(f"\n{'='*70}")
    print("RESULTADOS DEL BALANCE DE ENERGÍA")
    print(f"{'='*70}")
    print(f"\nCalor ENTRANTE por la base:")
    print(f"  - Conducción a nodos interiores:      {Q_in_conduction:.6f} W")
    print(f"  - Convección desde nodos base:        {Q_in_convection_base:.6f} W")
    print(f"  - TOTAL Q_in:                         {Q_in_total:.6f} W")
    print(f"\nCalor SALIENTE por convección (todas las superficies):")
    print(f"  Q_out = {Q_out_total:.6f} W")
    print(f"\nBalance:")
    print(f"  |Q_in - Q_out| = {error_abs:.6f} W")
    print(f"  Error relativo = {error_rel:.4f} %")
    print(f"\n  (Referencia: ver fin_fvm_1d.py líneas 108-120 para metodología)")

    # Criterio de aceptación
    if error_rel < 1.0:
        print(f"\n✓ BALANCE CORRECTO (error < 1%)")
        status = "CORRECTO"
    elif error_rel < 5.0:
        print(f"\n⚠ BALANCE ACEPTABLE (error < 5%)")
        status = "ACEPTABLE"
    else:
        print(f"\n✗ BALANCE INCORRECTO (error > 5%)")
        status = "INCORRECTO"

    # Estadísticas adicionales
    print(f"\n{'='*70}")
    print("ESTADÍSTICAS ADICIONALES")
    print(f"{'='*70}")
    print(f"Temperaturas:")
    print(f"  T_base:    {T_BASE:.2f} °C")
    print(f"  T_ambient: {T_INF:.2f} °C")
    print(f"  T_max en aleta: {np.max(T[center_nodes_list]):.2f} °C")
    print(f"  T_min en aleta: {np.min(T[center_nodes_list]):.2f} °C")
    print(f"  T_promedio:     {np.mean(T[center_nodes_list]):.2f} °C")

    # Área total expuesta
    total_boundary_area = sum(boundary_areas[c] for c in center_nodes_list)
    print(f"\nÁreas:")
    print(f"  Área superficial total: {total_boundary_area*1e6:.2f} mm²")
    print(f"  Área teórica (2*h*L + 2*t*L + t*h): {2*fin_height*fin_length*1e6 + 2*fin_thickness*fin_length*1e6 + fin_thickness*fin_height*1e6:.2f} mm²")

    # Flujo promedio
    avg_heat_flux_conv = Q_out_total / total_boundary_area if total_boundary_area > 0 else 0
    print(f"\nFlujos:")
    print(f"  Flujo de calor promedio (convección): {avg_heat_flux_conv:.2f} W/m²")

    print(f"\n{'='*70}\n")

    return {
        'Q_in': Q_in_total,
        'Q_out': Q_out_total,
        'error_abs': error_abs,
        'error_rel': error_rel,
        'status': status,
        'Q_in_details': Q_in_details,
        'Q_out_details': Q_out_details,
        'total_boundary_area': total_boundary_area,
        'avg_heat_flux': avg_heat_flux_conv,
        'T': T,
        'center_nodes': center_nodes_list,
        'coords': structural_coords
    }


if __name__ == "__main__":
    # Test con diferentes configuraciones

    print("\n\nTEST 1: Aleta rectangular simple (1mm espesor, 25mm altura)")
    result1 = verify_heat_balance(
        fin_thickness=0.001,
        fin_height=0.025,
        fin_length=0.05,
        tip_ratio=1.0,
        round_factor=0.0
    )

    print("\n\nTEST 2: Aleta más gruesa (10mm espesor, 25mm altura)")
    result2 = verify_heat_balance(
        fin_thickness=0.01,
        fin_height=0.025,
        fin_length=0.05,
        tip_ratio=1.0,
        round_factor=0.0
    )

    print("\n\nTEST 3: Aleta triangular (tip_ratio=0.5)")
    result3 = verify_heat_balance(
        fin_thickness=0.001,
        fin_height=0.025,
        fin_length=0.05,
        tip_ratio=0.5,
        round_factor=0.0
    )
