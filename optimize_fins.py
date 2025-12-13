# optimize_fins.py

from fin_2d_forward_euler import evaluate_fin_design
import matplotlib.pyplot as plt
import numpy as np
from grid_matrices_2d_with_centers import build_fin_grid_2d, plot_grid_with_blocks
import time

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

    # Determinar nHeight y nWidth (sabemos que usamos 11x11)
    nHeight = 11
    nWidth = 11

    # Graficar usando la función existente
    plot_grid_with_blocks(structural_coords, center_nodes, nHeight, nWidth, x_exaggeration=5.0)

    # Agregar título con información del diseño
    title = (f"Mejor Diseño de Aleta\n"
             f"Espesor: {best_design['thickness']*1000:.2f} mm, "
             f"Altura: {best_design['height']*1000:.1f} mm\n"
             f"tip_ratio: {best_design['tip_ratio']:.2f}, "
             f"round_factor: {best_design['round_factor']:.2f}\n"
             f"Masa total: {best_design['m_total']:.4f} kg")
    plt.gcf().suptitle(title, fontsize=12, y=0.98)

def optimize_fins(max_time=60):
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
                        gap=0.00001,  # 0.01mm gap for tighter packing
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
