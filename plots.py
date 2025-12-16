# plot_fins.py

from finEuler import solve_transient
import matplotlib.pyplot as plt
import numpy as np
from finGridwCenters import build_fin_grid_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from parameters import RHO, CP, K, H, T_INF, T_BASE, N_HEIGHT, N_WIDTH

def plot_best_fin(best_design, show=True):
    """
    Visualiza el mejor diseño de aleta encontrado.

    Parameters
    ----------
    best_design : dict
        Diccionario con los parámetros del mejor diseño
    show : bool
        Si True, muestra el gráfico. Si False, solo lo crea sin mostrarlo.
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
    plot_grid_with_blocks(structural_coords, center_nodes, N_HEIGHT, N_WIDTH, x_exaggeration=10.0)

    # Agregar título con información del diseño
    title = (f"Mejor Diseño de Aleta\n"
             f"Espesor: {best_design['thickness']*1000:.2f} mm, "
             f"Altura: {best_design['height']*1000:.1f} mm\n"
             f"tip_ratio: {best_design['tip_ratio']:.2f}, "
             f"round_factor: {best_design['round_factor']:.2f}\n"
             f"Masa total: {best_design['m_total']:.4f} kg")
    plt.gcf().suptitle(title, fontsize=12, y=0.98)

def plot_temperature_heatmap(best_design, show=True, verbose=True):
    """
    Visualiza la distribución de temperatura en la aleta óptima usando un heatmap.

    Parameters
    ----------
    best_design : dict
        Diccionario con los parámetros del mejor diseño
    show : bool
        Si True, muestra el gráfico. Si False, solo lo crea sin mostrarlo.
    verbose : bool
        Si True, imprime estadísticas de temperatura. Si False, las suprime.
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
    x_exag = 40.0  # Mayor exageración para aletas delgadas
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
    if show:
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
    if verbose:
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

def plot_temperature_3d(best_design, show=True):
    """
    Visualiza la aleta en 3D con distribución de temperatura.
    Muestra cómo el calor se disipa en la dirección Z (longitud de la aleta).

    Parameters
    ----------
    best_design : dict
        Diccionario con los parámetros del mejor diseño
    show : bool
        Si True, muestra el gráfico. Si False, solo lo crea sin mostrarlo.
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
    if show:
        plt.show()

    print("\nNOTA: La solución 2D asume temperatura uniforme en dirección Z.")
    print("Las caras frontal (Z=0) y trasera (Z=L) tienen la misma distribución de temperatura.")
    print("El calor se disipa principalmente:")
    print("  - Por las caras frontal y trasera (convección en Z)")
    print("  - Por los bordes laterales y superior (convección en X, Y)")


def plot_grid_with_blocks(structural_coords, center_nodes, nHeight, nWidth,
                          x_exaggeration=5.0):
    """
    Grafica la malla mostrando todos los nodos, nodos centrales y bloques.

    Parámetros
    ----------
    structural_coords : ndarray
        Coordenadas de todos los nodos en la malla
    center_nodes : list
        Lista de índices globales de los nodos centrales
    nHeight : int
        Número de nodos en dirección eta (filas)
    nWidth : int
        Número de nodos en dirección xi (columnas)
    x_exaggeration : float, optional
        Factor para estirar el eje X y hacer visible el espesor de la aleta
    """
    import matplotlib.patches as mpatches

    # Exagerar X solo para visualización (hacer visible el espesor de la aleta)
    scaled_coords = structural_coords.copy()
    scaled_coords[:, 0] *= x_exaggeration

    # Crear figura con mejor estilización
    plt.figure(figsize=(14, 10), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # Dibujar grilla primero (fondo sutil)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5,
            color='#cccccc', zorder=0)

    # Dibujar bloques con regiones rellenas para mejor visibilidad
    for center_idx in center_nodes:
        # Para un bloque 3×3, obtener fila/columna mín/máx del bloque
        center_row = center_idx // nWidth
        center_col = center_idx % nWidth

        # Determinar límites reales del bloque
        min_row = max(0, center_row - 1)
        max_row = min(nHeight - 1, center_row + 1)
        min_col = max(0, center_col - 1)
        max_col = min(nWidth - 1, center_col + 1)

        # Obtener las cuatro esquinas de la caja límite (orden antihorario)
        corners_idx = [
            min_row * nWidth + min_col,  # Inferior-izquierda
            min_row * nWidth + max_col,  # Inferior-derecha
            max_row * nWidth + max_col,  # Superior-derecha
            max_row * nWidth + min_col   # Superior-izquierda
        ]

        # Obtener coordenadas de las esquinas
        corners = [scaled_coords[idx] for idx in corners_idx]

        # Dibujar polígono relleno con color sutil
        polygon = Polygon(corners, fill=True, facecolor='#e3f2fd',
                         edgecolor='#1976d2', linewidth=2.0,
                         linestyle='-', alpha=0.4, zorder=1)
        ax.add_patch(polygon)

    # Graficar todos los nodos (más pequeños, sutiles)
    plt.scatter(scaled_coords[:, 0], scaled_coords[:, 1], s=35, marker='o',
               color='#64b5f6', edgecolors='#1976d2', linewidths=1.5,
               label='Nodos de la malla', zorder=3, alpha=0.8)

    # Graficar nodos centrales (más grandes, prominentes)
    center_coords = scaled_coords[center_nodes]
    plt.scatter(center_coords[:, 0], center_coords[:, 1], s=120, marker='o',
               color='#ff5722', edgecolors='#d84315', linewidths=2,
               label='Centros de volúmenes de control', zorder=5, alpha=0.95)

    # Agregar etiquetas solo para nodos centrales (menos desordenado)
    bubble_radius = 0.00035  # Radio fijo para todas las burbujas
    for i in center_nodes:
        x, y = scaled_coords[i]
        # Dibujar círculo de tamaño fijo centrado en el nodo
        circle = plt.Circle((x, y), bubble_radius,
                           facecolor='#d84315', edgecolor='none',
                           alpha=0.85, zorder=6)
        ax.add_patch(circle)
        # Agregar texto encima del círculo
        plt.text(x, y, str(i), fontsize=7, color='white',
                fontweight='bold', ha='center', va='center', zorder=7)

    # Estilización
    plt.title('Malla de Volúmenes Finitos con Bloques de Control',
             fontsize=18, fontweight='bold', pad=20, color='#263238')
    plt.xlabel(f'Dirección del Espesor de Aleta (X) [escalado ×{x_exaggeration}]',
              fontsize=14, fontweight='bold', color='#37474f')
    plt.ylabel('Dirección de Altura de Aleta (Y) [m]',
              fontsize=14, fontweight='bold', color='#37474f')

    # Agregar entrada personalizada a la leyenda para bloques
    block_patch = mpatches.Patch(facecolor='#e3f2fd', edgecolor='#1976d2',
                                 linewidth=2.0, alpha=0.4,
                                 label='Bloques de volúmenes de control')
    handles, labels_list = ax.get_legend_handles_labels()
    handles.append(block_patch)

    # Leyenda con mejor estilización
    legend = plt.legend(handles=handles, loc='upper right', fontsize=12,
                       frameon=True, fancybox=True, shadow=True,
                       framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#90a4ae')

    # Limpiar los ejes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#546e7a')
    ax.spines['bottom'].set_color('#546e7a')

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# ============================================================================
# EVOLUCIÓN TÉRMICA TRANSITORIA
# ============================================================================

def solve_and_track_evolution(k, h, rho, cp, T_inf, T_base,
                               dt=0.01, t_final=10.0, tol=1e-6,
                               fin_thickness=0.001, fin_height=0.025,
                               fin_length=0.05, tip_ratio=1.0, round_factor=0.0):
    """
    Resuelve el problema transitorio y guarda la evolución de temperaturas.

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
        Temperatura ambiente [K]
    T_base : float
        Temperatura de la base [K]
    dt : float
        Paso de tiempo inicial [s]
    t_final : float
        Tiempo final de simulación [s]
    tol : float
        Tolerancia de convergencia
    fin_thickness, fin_height, fin_length, tip_ratio, round_factor : float
        Parámetros geométricos de la aleta

    Returns
    -------
    tuple
        (time_history, T_avg_history, T_max_history, T_min_history, Q_history,
         T, coords, center_nodes, boundary_areas)
    """
    from finEuler import stable_timestep, forward_euler_step
    from parameters import GAP

    # Construir geometría
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

    # Inicializar temperatura
    T = np.ones(len(coords)) * T_inf

    # Identificar nodos de la base
    y_coords = coords[center_nodes, 1]
    y_min = y_coords.min()
    tol_geom = 1e-12 + 1e-6 * abs(y_min)
    bottom_cells = [c for c in center_nodes if coords[c, 1] <= y_min + tol_geom]

    # Aplicar temperatura de base
    for c in bottom_cells:
        T[c] = T_base

    # Verificar estabilidad
    dt_stable = stable_timestep(center_nodes, neighbours_dict, face_areas,
                                center_distances, boundary_areas, volumes,
                                rho, cp, k, h, safety=0.5)
    if dt_stable is not None and dt > dt_stable:
        print(f"Usando dt={dt_stable:.4e} para estabilidad")
        dt = dt_stable

    # Listas para guardar evolución
    time_history = [0.0]
    T_avg_history = [np.mean(T[center_nodes])]
    T_max_history = [np.max(T[center_nodes])]
    T_min_history = [np.min(T[center_nodes])]

    # Calcular calor disipado inicial
    Q_initial = sum(h * boundary_areas[node] * (T[node] - T_inf) for node in center_nodes)
    Q_history = [Q_initial]

    time = 0.0
    nsteps = int(np.ceil(t_final / dt))

    print(f"\nSimulando evolución térmica...")
    print(f"  dt = {dt:.6f} s, t_final = {t_final} s")

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

        # Guardar datos cada cierto número de pasos
        if n % 10 == 0 or n == nsteps - 1:
            time_history.append(time)
            T_avg_history.append(np.mean(T[center_nodes]))
            T_max_history.append(np.max(T[center_nodes]))
            T_min_history.append(np.min(T[center_nodes]))

            # Calcular calor disipado por convección
            Q_dissipated = sum(h * boundary_areas[node] * (T[node] - T_inf) for node in center_nodes)
            Q_history.append(Q_dissipated)

        # Verificar convergencia
        max_diff = np.max(np.abs(T - T_old))
        if max_diff < tol:
            print(f"  Convergencia en t = {time:.4f} s (paso {n})")
            break

    return (time_history, T_avg_history, T_max_history, T_min_history, Q_history,
            T, coords, center_nodes, boundary_areas)


def plot_thermal_evolution(time_history, T_avg, T_max, T_min, Q_history, config, n_fins):
    """
    Grafica la evolución de las temperaturas y el calor disipado en función del tiempo
    para el disipador completo con todas las aletas.

    Parameters
    ----------
    time_history : list
        Historia temporal [s]
    T_avg, T_max, T_min : list
        Temperaturas promedio, máxima y mínima [°C]
    Q_history : list
        Historia de calor disipado por UNA aleta [W]
    config : dict
        Configuración de la aleta
    n_fins : int
        Número de aletas en el disipador

    Returns
    -------
    tuple
        (fig, (ax1, ax2)) - Figura y ejes de matplotlib
    """
    from parameters import Q_REQUIRED

    # Calcular calor total del disipador (todas las aletas)
    Q_total_history = [Q * n_fins for Q in Q_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1: Temperaturas
    ax1.plot(time_history, T_max, 'r-', linewidth=2, label='T máxima')
    ax1.plot(time_history, T_avg, 'b-', linewidth=2, label='T promedio')
    ax1.plot(time_history, T_min, 'g-', linewidth=2, label='T mínima')

    ax1.axhline(y=T_BASE, color='r', linestyle='--', alpha=0.5, label=f'T_base = {T_BASE}°C')
    ax1.axhline(y=T_INF, color='b', linestyle='--', alpha=0.5, label=f'T_inf = {T_INF}°C')

    ax1.set_xlabel('Tiempo [s]', fontsize=12)
    ax1.set_ylabel('Temperatura [°C]', fontsize=12)
    ax1.set_title('Evolución Térmica hasta Estado Estacionario', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='right')
    ax1.grid(True, alpha=0.3)

    # Agregar información de la configuración
    info_text = (f"Configuración del disipador:\n"
                 f"  Espesor aleta: {config['thickness']*1000:.1f} mm\n"
                 f"  Altura aleta: {config['height']*1000:.1f} mm\n"
                 f"  Longitud: {config['length']*1000:.1f} mm\n"
                 f"  Número de aletas: {n_fins}")
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 2: Calor disipado por el DISIPADOR COMPLETO
    ax2.plot(time_history, Q_total_history, 'm-', linewidth=2.5, label=f'Disipador completo ({n_fins} aletas)')
    ax2.axhline(y=Q_REQUIRED, color='r', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Q requerido = {Q_REQUIRED} W')

    ax2.set_xlabel('Tiempo [s]', fontsize=12)
    ax2.set_ylabel('Calor Disipado [W]', fontsize=12)
    ax2.set_title('Evolución del Calor Disipado - Disipador Completo', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='right')
    ax2.grid(True, alpha=0.3)

    # Agregar valor final y comparación con requerimiento
    Q_final_total = Q_total_history[-1]
    deficit = Q_REQUIRED - Q_final_total

    if deficit > 0:
        status_color = 'orange'
        status_text = f'Q_total = {Q_final_total:.2f} W\nDéficit = {deficit:.2f} W\n⚠ INSUFICIENTE'
    else:
        status_color = 'lightgreen'
        surplus = Q_final_total - Q_REQUIRED
        status_text = f'Q_total = {Q_final_total:.2f} W\nExcedente = {surplus:.2f} W\n✓ SUFICIENTE'

    ax2.text(0.98, 0.98, status_text,
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))

    plt.tight_layout()
    return fig, (ax1, ax2)


# ============================================================================
# VISUALIZACIÓN DEL DISIPADOR COMPLETO CON TODAS LAS ALETAS
# ============================================================================

def plot_complete_heatsink(fin_thickness, fin_height, fin_length, n_fins, gap,
                          tip_ratio=1.0, round_factor=0.0):
    """
    Visualiza el disipador completo con todas las aletas distribuidas espacialmente.

    Parameters
    ----------
    fin_thickness : float
        Espesor de cada aleta [m]
    fin_height : float
        Altura de las aletas [m]
    fin_length : float
        Longitud de las aletas [m]
    n_fins : int
        Número de aletas en el disipador
    gap : float
        Espacio entre aletas [m]
    tip_ratio : float
        Relación de ahusamiento
    round_factor : float
        Factor de redondeo

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada
    """
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches

    # Calcular dimensiones del disipador
    pitch = fin_thickness + gap
    total_width = n_fins * pitch

    # Crear figura con dos vistas
    fig = plt.figure(figsize=(18, 8))

    # ============ VISTA SUPERIOR (X-Z) ============
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor('#f5f5f5')

    # Dibujar base del disipador
    base = Rectangle((0, 0), total_width, fin_length,
                    facecolor='#90a4ae', edgecolor='black',
                    linewidth=2, alpha=0.3, label='Base del disipador')
    ax1.add_patch(base)

    # Dibujar cada aleta
    for i in range(n_fins):
        x_pos = i * pitch + gap/2  # Centrar aleta en su espacio

        # Aleta como rectángulo en vista superior
        fin_rect = Rectangle((x_pos, 0), fin_thickness, fin_length,
                            facecolor='#ff6b6b', edgecolor='#c92a2a',
                            linewidth=1.5, alpha=0.8)
        ax1.add_patch(fin_rect)

        # Numerar algunas aletas para referencia
        if i % 5 == 0 or i == n_fins - 1:
            ax1.text(x_pos + fin_thickness/2, fin_length/2, str(i+1),
                    ha='center', va='center', fontsize=6,
                    color='white', fontweight='bold')

    ax1.set_xlim(-gap, total_width + gap)
    ax1.set_ylim(-0.005, fin_length + 0.005)
    ax1.set_xlabel('Ancho del disipador [m]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Longitud de aleta (Z) [m]', fontsize=12, fontweight='bold')
    ax1.set_title('Vista Superior (Plano X-Z)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')

    # Información
    info_text = (f"Configuración:\n"
                f"  Aletas: {n_fins}\n"
                f"  Espesor: {fin_thickness*1000:.3f} mm\n"
                f"  Gap: {gap*1000:.1f} mm\n"
                f"  Pitch: {pitch*1000:.2f} mm\n"
                f"  Ancho total: {total_width*1000:.1f} mm")
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    # ============ VISTA FRONTAL (X-Y) ============
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor('#f5f5f5')

    # Dibujar base
    base_height = 0.003  # 3mm de altura de base
    base_front = Rectangle((0, 0), total_width, base_height,
                          facecolor='#90a4ae', edgecolor='black',
                          linewidth=2, alpha=0.5, label='Base')
    ax2.add_patch(base_front)

    # Dibujar perfil de cada aleta
    for i in range(n_fins):
        x_pos = i * pitch + gap/2

        # Perfil de aleta (ahusado si tip_ratio < 1.0)
        if tip_ratio < 1.0:
            # Aleta ahusada - trapecio
            tip_thickness = fin_thickness * tip_ratio
            x_left_base = x_pos
            x_right_base = x_pos + fin_thickness
            x_left_tip = x_pos + (fin_thickness - tip_thickness)/2
            x_right_tip = x_pos + fin_thickness - (fin_thickness - tip_thickness)/2

            trapezoid = mpatches.Polygon([
                [x_left_base, base_height],
                [x_right_base, base_height],
                [x_right_tip, base_height + fin_height],
                [x_left_tip, base_height + fin_height]
            ], facecolor='#ff6b6b', edgecolor='#c92a2a',
               linewidth=1.5, alpha=0.8)
            ax2.add_patch(trapezoid)
        else:
            # Aleta rectangular
            fin_front = Rectangle((x_pos, base_height), fin_thickness, fin_height,
                                facecolor='#ff6b6b', edgecolor='#c92a2a',
                                linewidth=1.5, alpha=0.8)
            ax2.add_patch(fin_front)

    ax2.set_xlim(-gap, total_width + gap)
    ax2.set_ylim(-0.001, base_height + fin_height + 0.005)
    ax2.set_xlabel('Ancho del disipador [m]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Altura (Y) [m]', fontsize=12, fontweight='bold')
    ax2.set_title('Vista Frontal (Plano X-Y)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')

    # Cotas de dimensión
    ax2.annotate('', xy=(0, base_height + fin_height + 0.003),
                xytext=(0, base_height),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax2.text(-0.002, base_height + fin_height/2,
            f'{fin_height*1000:.1f} mm',
            rotation=90, va='center', ha='right',
            fontsize=10, color='blue', fontweight='bold')

    # Título general
    fig.suptitle(f'Disipador Completo - {n_fins} Aletas',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig
