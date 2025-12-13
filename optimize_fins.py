# optimize_fins.py

from fin_2d_forward_euler import evaluate_fin_design, solve_transient
import matplotlib.pyplot as plt
import numpy as np
from grid_matrices_2d_with_centers import build_fin_grid_2d, plot_grid_with_blocks
import time
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from parameters import RHO, CP, K, H, T_INF, T_BASE, N_HEIGHT, N_WIDTH

def plot_best_fin(best_design):
    """
    Visualiza el mejor diseño de aleta encontrado.

    Parameters
    ----------
    best_design : dict
        Diccionario con los parámetros del mejor diseño
    """
    if best_design is None:
        print("No hay diseño para graficar.")
        return

    # Construir la grilla con los parámetros óptimos
    (structural_coords, center_nodes, neighbours_dict, areas_dict,
     volumes_dict, blocks, center_distances, face_lengths,
     face_areas, boundary_areas) = build_fin_grid_2d(
        fin_thickness=best_design['thickness'],
        fin_height=best_design['height'],
        fin_length=0.05,
        tip_ratio=best_design['tip_ratio'],
        round_factor=best_design['round_factor']
    )

    # Graficar usando la función existente
    plot_grid_with_blocks(structural_coords, center_nodes, N_HEIGHT, N_WIDTH, x_exaggeration=5.0)

    # Agregar título con información del diseño
    title = (f"Mejor Diseño de Aleta\n"
             f"Espesor: {best_design['thickness']*1000:.2f} mm, "
             f"Altura: {best_design['height']*1000:.1f} mm\n"
             f"tip_ratio: {best_design['tip_ratio']:.2f}, "
             f"round_factor: {best_design['round_factor']:.2f}\n"
             f"Masa total: {best_design['m_total']:.4f} kg")
    plt.gcf().suptitle(title, fontsize=12, y=0.98)

def plot_temperature_heatmap(best_design):
    """
    Visualiza la distribución de temperatura en la aleta óptima usando un heatmap.

    Parameters
    ----------
    best_design : dict
        Diccionario con los parámetros del mejor diseño
    """
    if best_design is None:
        print("No hay diseño para graficar.")
        return

    # Resolver el campo de temperaturas
    T, coords, center_nodes = solve_transient(
        k=K, h=H, rho=RHO, cp=CP,
        T_inf=T_INF,
        T_base=T_BASE,
        dt=0.005,
        t_final=5.0,
        tol=1e-6,
        fin_thickness=best_design['thickness'],
        fin_height=best_design['height'],
        fin_length=0.05,
        tip_ratio=best_design['tip_ratio'],
        round_factor=best_design['round_factor']
    )

    # Construir la geometría
    (structural_coords, center_nodes_list, neighbours_dict, areas_dict,
     volumes_dict, blocks, *_) = build_fin_grid_2d(
        fin_thickness=best_design['thickness'],
        fin_height=best_design['height'],
        fin_length=0.05,
        tip_ratio=best_design['tip_ratio'],
        round_factor=best_design['round_factor']
    )

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Heatmap de bloques (control volumes) ---
    x_exag = 5.0
    scaled_coords = structural_coords.copy()
    scaled_coords[:, 0] *= x_exag

    patches = []
    colors = []

    # Para cada centro de control volume, dibujar su bloque con color según temperatura
    for center_idx in center_nodes_list:
        center_row = center_idx // N_WIDTH
        center_col = center_idx % N_WIDTH

        # Determinar límites del bloque
        min_row = max(0, center_row - 1)
        max_row = min(N_HEIGHT - 1, center_row + 1)
        min_col = max(0, center_col - 1)
        max_col = min(N_WIDTH - 1, center_col + 1)

        # Esquinas del bloque
        corners_idx = [
            min_row * N_WIDTH + min_col,  # bottom-left
            min_row * N_WIDTH + max_col,  # bottom-right
            max_row * N_WIDTH + max_col,  # top-right
            max_row * N_WIDTH + min_col   # top-left
        ]

        corners = [scaled_coords[idx] for idx in corners_idx]
        polygon = Polygon(corners, closed=True)
        patches.append(polygon)
        colors.append(T[center_idx])

    # Crear colección de patches con colormap invertido (hot → frío=blanco, caliente=rojo/negro)
    p = PatchCollection(patches, cmap='hot_r', edgecolors='black', linewidths=0.5, alpha=0.9)
    p.set_array(np.array(colors))
    p.set_clim(T_INF, T_BASE)
    ax1.add_collection(p)

    # Configurar ejes
    ax1.set_xlim(scaled_coords[:, 0].min() - 0.001, scaled_coords[:, 0].max() + 0.001)
    ax1.set_ylim(scaled_coords[:, 1].min() - 0.001, scaled_coords[:, 1].max() + 0.001)
    ax1.set_aspect('equal')
    ax1.set_xlabel(f'Espesor (X) [exagerado x{x_exag}]')
    ax1.set_ylabel('Altura (Y) [m]')
    ax1.set_title('Distribución de Temperatura - Control Volumes')

    # Colorbar
    cbar1 = plt.colorbar(p, ax=ax1)
    cbar1.set_label('Temperatura [°C]')

    # --- Plot 2: Interpolación suave de temperatura ---
    # Usar scatter con interpolación para visualización más suave
    center_coords_scaled = scaled_coords[center_nodes_list]
    center_temps = T[center_nodes_list]

    scatter = ax2.scatter(center_coords_scaled[:, 0], center_coords_scaled[:, 1],
                         c=center_temps, cmap='hot_r', s=100,
                         vmin=T_INF, vmax=T_BASE, edgecolors='black', linewidths=0.5)

    # Dibujar contorno de la aleta
    ax2.plot(scaled_coords[:, 0], scaled_coords[:, 1], 'k.', markersize=1, alpha=0.3)

    ax2.set_xlim(scaled_coords[:, 0].min() - 0.001, scaled_coords[:, 0].max() + 0.001)
    ax2.set_ylim(scaled_coords[:, 1].min() - 0.001, scaled_coords[:, 1].max() + 0.001)
    ax2.set_aspect('equal')
    ax2.set_xlabel(f'Espesor (X) [exagerado x{x_exag}]')
    ax2.set_ylabel('Altura (Y) [m]')
    ax2.set_title('Distribución de Temperatura - Centros de CV')

    # Colorbar
    cbar2 = plt.colorbar(scatter, ax=ax2)
    cbar2.set_label('Temperatura [°C]')

    # Título general
    title = (f"Análisis Térmico del Mejor Diseño\n"
             f"t={best_design['thickness']*1000:.2f}mm, "
             f"h={best_design['height']*1000:.1f}mm, "
             f"tip_ratio={best_design['tip_ratio']:.2f}, "
             f"round_factor={best_design['round_factor']:.2f}")
    fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()
    plt.show()

    # Análisis: nodos centrales vs nodos de borde (en dirección del espesor)
    # Nodos centrales: solo convección en caras front/back (Z)
    # Nodos de borde: convección en front/back + lados (X boundaries)
    x_coords = structural_coords[center_nodes_list, 0]
    x_threshold = best_design['thickness'] * 0.3

    center_node_temps = []
    edge_node_temps = []

    for idx in center_nodes_list:
        x_dist = abs(structural_coords[idx, 0])
        if x_dist < x_threshold:
            center_node_temps.append(T[idx])
        else:
            edge_node_temps.append(T[idx])

    # Estadísticas de temperatura
    print("\nEstadísticas de temperatura:")
    print(f"  T_base (fija):    {T_BASE:.2f} °C")
    print(f"  T_ambiente:       {T_INF:.2f} °C")
    print(f"  T_max en aleta:   {np.max(T[center_nodes_list]):.2f} °C")
    print(f"  T_min en aleta:   {np.min(T[center_nodes_list]):.2f} °C")
    print(f"  T_promedio:       {np.mean(T[center_nodes_list]):.2f} °C")
    print()
    print("Gradiente térmico en espesor (centro vs borde lateral):")
    if center_node_temps and edge_node_temps:
        print(f"  T_promedio nodos centrales:      {np.mean(center_node_temps):.2f} °C")
        print(f"  T_promedio nodos borde lateral:  {np.mean(edge_node_temps):.2f} °C")
        print(f"  Diferencia (centro - borde):     {np.mean(center_node_temps) - np.mean(edge_node_temps):.2f} °C")

def plot_temperature_3d(best_design):
    """
    Visualiza la aleta en 3D con distribución de temperatura.
    Muestra cómo el calor se disipa en la dirección Z (longitud de la aleta).

    Parameters
    ----------
    best_design : dict
        Diccionario con los parámetros del mejor diseño
    """
    if best_design is None:
        print("No hay diseño para graficar.")
        return

    fin_length = 0.05

    # Resolver el campo de temperaturas (2D)
    T, coords_2d, center_nodes = solve_transient(
        k=K, h=H, rho=RHO, cp=CP,
        T_inf=T_INF,
        T_base=T_BASE,
        dt=0.005,
        t_final=5.0,
        tol=1e-6,
        fin_thickness=best_design['thickness'],
        fin_height=best_design['height'],
        fin_length=fin_length,
        tip_ratio=best_design['tip_ratio'],
        round_factor=best_design['round_factor']
    )

    # Construir la geometría 2D
    (structural_coords_2d, center_nodes_list, neighbours_dict, areas_dict,
     volumes_dict, blocks, *_) = build_fin_grid_2d(
        fin_thickness=best_design['thickness'],
        fin_height=best_design['height'],
        fin_length=fin_length,
        tip_ratio=best_design['tip_ratio'],
        round_factor=best_design['round_factor']
    )

    # Crear figura 3D
    fig = plt.figure(figsize=(18, 8))

    # --- Plot 1: Vista 3D con superficie frontal ---
    ax1 = fig.add_subplot(121, projection='3d')

    # Crear polígonos 3D para la superficie frontal (Z=0) y trasera (Z=fin_length)
    polygons_front = []
    polygons_back = []
    colors = []

    for center_idx in center_nodes_list:
        center_row = center_idx // N_WIDTH
        center_col = center_idx % N_WIDTH

        # Determinar límites del bloque
        min_row = max(0, center_row - 1)
        max_row = min(N_HEIGHT - 1, center_row + 1)
        min_col = max(0, center_col - 1)
        max_col = min(N_WIDTH - 1, center_col + 1)

        # Esquinas del bloque en 2D
        corners_idx = [
            min_row * N_WIDTH + min_col,  # bottom-left
            min_row * N_WIDTH + max_col,  # bottom-right
            max_row * N_WIDTH + max_col,  # top-right
            max_row * N_WIDTH + min_col   # top-left
        ]

        # Crear vértices 3D para superficie frontal (Z=0)
        verts_front = []
        for idx in corners_idx:
            x, y = structural_coords_2d[idx]
            verts_front.append([x, y, 0.0])

        # Crear vértices 3D para superficie trasera (Z=fin_length)
        verts_back = []
        for idx in corners_idx:
            x, y = structural_coords_2d[idx]
            verts_back.append([x, y, fin_length])

        polygons_front.append(verts_front)
        polygons_back.append(verts_back)
        colors.append(T[center_idx])

    # Normalizar colores
    colors_norm = np.array(colors)
    colors_normalized = (colors_norm - T_INF) / (T_BASE - T_INF)

    # Crear colección 3D para superficies frontal y trasera
    poly_front = Poly3DCollection(polygons_front, alpha=0.8, edgecolors='black', linewidths=0.3)
    poly_back = Poly3DCollection(polygons_back, alpha=0.8, edgecolors='black', linewidths=0.3)

    # Aplicar colormap (hot_r)
    cmap = plt.cm.hot_r
    poly_front.set_facecolors(cmap(colors_normalized))
    poly_back.set_facecolors(cmap(colors_normalized))

    ax1.add_collection3d(poly_front)
    ax1.add_collection3d(poly_back)

    # Dibujar los lados (conectar frontal con trasera) para bloques en el perímetro
    side_polygons = []
    side_colors = []

    for center_idx in center_nodes_list:
        center_row = center_idx // N_WIDTH
        center_col = center_idx % N_WIDTH

        min_row = max(0, center_row - 1)
        max_row = min(N_HEIGHT - 1, center_row + 1)
        min_col = max(0, center_col - 1)
        max_col = min(N_WIDTH - 1, center_col + 1)

        corners_idx = [
            min_row * N_WIDTH + min_col,
            min_row * N_WIDTH + max_col,
            max_row * N_WIDTH + max_col,
            max_row * N_WIDTH + min_col
        ]

        # Para cada arista del bloque, verificar si está en el perímetro
        for i in range(4):
            j = (i + 1) % 4
            idx1, idx2 = corners_idx[i], corners_idx[j]

            x1, y1 = structural_coords_2d[idx1]
            x2, y2 = structural_coords_2d[idx2]

            # Verificar si esta arista está en el borde del fin
            is_boundary = False

            # Bordes laterales (X boundaries)
            if abs(x1 - structural_coords_2d[:, 0].min()) < 1e-6 and abs(x2 - structural_coords_2d[:, 0].min()) < 1e-6:
                is_boundary = True
            if abs(x1 - structural_coords_2d[:, 0].max()) < 1e-6 and abs(x2 - structural_coords_2d[:, 0].max()) < 1e-6:
                is_boundary = True

            # Borde superior (Y max)
            if abs(y1 - structural_coords_2d[:, 1].max()) < 1e-6 and abs(y2 - structural_coords_2d[:, 1].max()) < 1e-6:
                is_boundary = True

            if is_boundary:
                # Crear polígono lateral
                verts_side = [
                    [x1, y1, 0.0],
                    [x2, y2, 0.0],
                    [x2, y2, fin_length],
                    [x1, y1, fin_length]
                ]
                side_polygons.append(verts_side)
                side_colors.append(T[center_idx])

    if side_polygons:
        side_colors_norm = np.array(side_colors)
        side_colors_normalized = (side_colors_norm - T_INF) / (T_BASE - T_INF)

        poly_sides = Poly3DCollection(side_polygons, alpha=0.9, edgecolors='black', linewidths=0.3)
        poly_sides.set_facecolors(cmap(side_colors_normalized))
        ax1.add_collection3d(poly_sides)

    # Configurar ejes
    ax1.set_xlabel('Espesor X [m]')
    ax1.set_ylabel('Altura Y [m]')
    ax1.set_zlabel('Longitud Z [m]')
    ax1.set_title('Vista 3D - Distribución de Temperatura')

    # Límites de ejes
    ax1.set_xlim(structural_coords_2d[:, 0].min(), structural_coords_2d[:, 0].max())
    ax1.set_ylim(structural_coords_2d[:, 1].min(), structural_coords_2d[:, 1].max())
    ax1.set_zlim(0, fin_length)

    # --- Plot 2: Vista desde arriba (mirando -Y) para ver disipación en Z ---
    ax2 = fig.add_subplot(122, projection='3d')

    # Mismo código pero con vista diferente
    poly_front2 = Poly3DCollection(polygons_front, alpha=0.8, edgecolors='black', linewidths=0.3)
    poly_back2 = Poly3DCollection(polygons_back, alpha=0.8, edgecolors='black', linewidths=0.3)
    poly_front2.set_facecolors(cmap(colors_normalized))
    poly_back2.set_facecolors(cmap(colors_normalized))

    ax2.add_collection3d(poly_front2)
    ax2.add_collection3d(poly_back2)

    if side_polygons:
        poly_sides2 = Poly3DCollection(side_polygons, alpha=0.9, edgecolors='black', linewidths=0.3)
        poly_sides2.set_facecolors(cmap(side_colors_normalized))
        ax2.add_collection3d(poly_sides2)

    ax2.set_xlabel('Espesor X [m]')
    ax2.set_ylabel('Altura Y [m]')
    ax2.set_zlabel('Longitud Z [m]')
    ax2.set_title('Vista Superior - Disipación a lo largo de Z')

    ax2.set_xlim(structural_coords_2d[:, 0].min(), structural_coords_2d[:, 0].max())
    ax2.set_ylim(structural_coords_2d[:, 1].min(), structural_coords_2d[:, 1].max())
    ax2.set_zlim(0, fin_length)

    # Cambiar ángulo de vista para ver desde arriba
    ax2.view_init(elev=10, azim=0)

    # Título general
    title = (f"Visualización 3D - Disipación de Calor\n"
             f"t={best_design['thickness']*1000:.2f}mm, "
             f"h={best_design['height']*1000:.1f}mm, "
             f"L={fin_length*1000:.1f}mm")
    fig.suptitle(title, fontsize=14, y=0.95)

    # Añadir colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=T_INF, vmax=T_BASE))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('Temperatura [°C]', fontsize=12)

    plt.tight_layout()
    plt.show()

    print("\nNOTA: La solución 2D asume temperatura uniforme en dirección Z.")
    print("Las caras frontal (Z=0) y trasera (Z=L) tienen la misma distribución de temperatura.")
    print("El calor se disipa principalmente:")
    print("  - Por las caras frontal y trasera (convección en Z)")
    print("  - Por los bordes laterales y superior (convección en X, Y)")

def optimize_fins(max_time=600):
    """
    Optimiza el diseño barriendo:
    - thickness
    - height
    - tip_ratio (punta más o menos angosta)
    - round_factor (lados más o menos redondeados)

    Parameters
    ----------
    max_time : float
        Tiempo máximo de optimización en segundos (por defecto 60s)
    """

    start_time = time.time()
    best = None
    time_exceeded = False

    # POCOS valores para que corra rápido
    thickness_values = [0.001, 0.0012]      # 1.0 mm y 1.2 mm (reduced for feasibility)
    height_values    = [0.025]        #25 mm

    # Forma: parámetro 1 → angostura de la punta
    tip_ratios       = [1.0, 0.7, 0.5]       # 1.0 rectangular, 0.7 y 0.5 más angosta

    # Forma: parámetro 2 → redondez
    round_factors    = [0.0, 0.4, 0.8]       # 0 = lados rectos, 0.8 = bien redondeada

    for t in thickness_values:
        for h in height_values:
            for tr in tip_ratios:
                for rf in round_factors:
                    # Chequear si se excedió el tiempo
                    if time.time() - start_time > max_time:
                        print(f"\n¡Tiempo máximo excedido ({max_time}s)! Deteniendo optimización...")
                        time_exceeded = True
                        break

                    print(f"\nProbando diseño:")
                    print(f"  thickness   = {t*1000:.2f} mm")
                    print(f"  height      = {h*1000:.1f} mm")
                    print(f"  tip_ratio   = {tr:.2f}")
                    print(f"  round_fact  = {rf:.2f}")

                    res = evaluate_fin_design(
                        fin_thickness=t,
                        fin_height=h,
                        fin_length=0.05,
                        gap=0.001,  # 0.01mm gap for tighter packing
                        tip_ratio=tr,
                        round_factor=rf
                    )

                    if (res is None) or (not res["feasible"]):
                        print("  → no factible")
                        continue

                    print(f"  → factible, masa total = {res['m_total']:.4f} kg")

                    if (best is None) or (res["m_total"] < best["m_total"]):
                        best = res
                        print("  *** nuevo mejor diseño ***")

                if time_exceeded:
                    break
            if time_exceeded:
                break
        if time_exceeded:
            break

    elapsed_time = time.time() - start_time
    return best, elapsed_time


if __name__ == "__main__":
    best, elapsed_time = optimize_fins()

    print("\n========================")
    print("       MEJOR DISEÑO     ")
    print("========================")
    print(f"Tiempo de optimización: {elapsed_time:.2f} segundos")
    print()

    if best is None:
        print("No se encontró ningún diseño factible.")
    else:
        print(f"Espesor (t):       {best['thickness']*1000:.2f} mm")
        print(f"Altura (h):        {best['height']*1000:.1f} mm")
        print(f"tip_ratio:         {best['tip_ratio']:.2f}")
        print(f"round_factor:      {best['round_factor']:.2f}")
        print(f"Q por aleta:       {best['Q_fin']:.3f} W")
        print(f"Masa por aleta:    {best['m_fin']:.6f} kg")
        print(f"Aletas requeridas: {best['n_required']:.1f}")
        print(f"Aletas máximas:    {best['n_max']}")
        print(f"Masa total aletas: {best['m_total']:.4f} kg")

        # Graficar el mejor diseño
        print("\nGenerando gráfico del mejor diseño...")
        plot_best_fin(best)

        # Graficar distribución de temperaturas
        print("\nGenerando heatmap de temperaturas...")
        plot_temperature_heatmap(best)
