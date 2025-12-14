"""
Gráfico de evolución térmica desde el estado inicial hasta el estado estacionario.
"""

import numpy as np
import matplotlib.pyplot as plt

from grid_matrices_2d_with_centers import build_fin_grid_2d
from fin_2d_forward_euler import stable_timestep, forward_euler_step
from parameters import RHO, CP, K, H, T_INF, T_BASE


def solve_and_track_evolution(k, h, rho, cp, T_inf, T_base,
                               dt=0.01, t_final=10.0, tol=1e-6,
                               fin_thickness=0.001, fin_height=0.025,
                               fin_length=0.05, tip_ratio=1.0, round_factor=0.0):
    """
    Resuelve el problema transitorio y guarda la evolución de temperaturas.
    """

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


def calculate_heat_dissipated(T, center_nodes, boundary_areas, h, T_inf):
    """
    Calcula el calor total disipado por convección en el estado estacionario.

    Q_total = Σ h * A_surf * (T - T_inf)
    """
    Q_total = 0.0
    for node in center_nodes:
        A_surf = boundary_areas[node]
        Q_conv = h * A_surf * (T[node] - T_inf)
        Q_total += Q_conv

    return Q_total


def plot_thermal_evolution(time_history, T_avg, T_max, T_min, Q_history, config, n_fins):
    """
    Grafica la evolución de las temperaturas y el calor disipado en función del tiempo
    para el disipador completo con todas las aletas.
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


if __name__ == "__main__":
    # Configuración de la aleta
    config = {
        'thickness': 0.001,   # 1 mm
        'height': 0.025,      # 25 mm
        'length': 0.05,       # 50 mm
        'tip_ratio': 1.0,     # Rectangular
        'round_factor': 0.0   # Sin redondeo
    }

    # Ejecutar simulación
    (time_hist, T_avg, T_max, T_min, Q_hist, T_final, coords, centers, boundary_areas) = solve_and_track_evolution(
        k=K, h=H, rho=RHO, cp=CP,
        T_inf=T_INF, T_base=T_BASE,
        dt=0.01, t_final=10.0, tol=1e-6,
        fin_thickness=config['thickness'],
        fin_height=config['height'],
        fin_length=config['length'],
        tip_ratio=config['tip_ratio'],
        round_factor=config['round_factor']
    )

    # Calcular calor disipado por UNA aleta (estado final)
    Q_per_fin = Q_hist[-1]  # Usar el último valor del historial

    # Calcular cuántas aletas se necesitan para 500W
    from parameters import Q_REQUIRED, BASE_SIZE
    n_fins_needed = Q_REQUIRED / Q_per_fin if Q_per_fin > 0 else float('inf')

    # Calcular espaciamiento entre aletas
    gap = config['thickness']  # Asumir gap igual al espesor
    total_thickness_per_fin = config['thickness'] + gap
    n_fins_fit = int(BASE_SIZE / total_thickness_per_fin)

    # Graficar el disipador completo con todas las aletas que caben
    fig, axes = plot_thermal_evolution(time_hist, T_avg, T_max, T_min, Q_hist, config, n_fins_fit)

    # Guardar
    fig.savefig('thermal_evolution.png', dpi=150, bbox_inches='tight')
    print(f"\nGráfico guardado: thermal_evolution.png")

    # Calcular calor total del disipador
    Q_total_dissipated = Q_per_fin * n_fins_fit

    # Mostrar estadísticas finales
    print(f"\n{'='*70}")
    print("ESTADO ESTACIONARIO")
    print(f"{'='*70}")
    print(f"\nTemperaturas:")
    print(f"  T_max = {T_max[-1]:.2f} °C")
    print(f"  T_avg = {T_avg[-1]:.2f} °C")
    print(f"  T_min = {T_min[-1]:.2f} °C")
    print(f"  Tiempo de simulación: {time_hist[-1]:.4f} s")

    print(f"\n{'='*70}")
    print("CALOR DISIPADO - UNA ALETA")
    print(f"{'='*70}")
    print(f"  Q_por_aleta = {Q_per_fin:.4f} W")

    print(f"\n{'='*70}")
    print("CALOR DISIPADO - DISIPADOR COMPLETO")
    print(f"{'='*70}")
    print(f"\nConfiguración:")
    print(f"  Aletas que caben (gap={gap*1000:.1f}mm): {n_fins_fit}")
    print(f"  Q_total disipado: {Q_total_dissipated:.2f} W")

    print(f"\nComparación con requerimiento:")
    print(f"  Q_requerido: {Q_REQUIRED} W")
    print(f"  Q_total: {Q_total_dissipated:.2f} W")

    deficit = Q_REQUIRED - Q_total_dissipated
    if deficit > 0:
        print(f"  Déficit: {deficit:.2f} W")
        print(f"\n✗ DISEÑO INSUFICIENTE")
        print(f"  Se necesitan {n_fins_needed:.1f} aletas para cumplir el requerimiento")
    else:
        surplus = Q_total_dissipated - Q_REQUIRED
        print(f"  Excedente: {surplus:.2f} W")
        print(f"\n✓ DISEÑO SUFICIENTE")

    plt.show()
