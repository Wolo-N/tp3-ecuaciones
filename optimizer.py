# optimize_fins.py

from fin_2d_forward_euler import evaluate_fin_design
from plots import plot_best_fin
import time
import json
import os
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar sin mostrar
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def save_optimal_design(design, iteration_number, output_dir="optimization_results"):
    """
    Guarda el diseño óptimo como PNG y JSON.

    Parameters
    ----------
    design : dict
        Diccionario con los parámetros del diseño
    iteration_number : int
        Número de iteración del diseño óptimo
    output_dir : str
        Directorio donde guardar los resultados
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp para nombres únicos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Nombre base para archivos
    base_name = f"optimal_{iteration_number:03d}_{timestamp}"

    # 1. Guardar parámetros en JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, 'w') as f:
        json.dump(design, f, indent=2)

    # 2. Guardar solo geometría de la aleta (sin mostrar)
    fig = plt.figure(figsize=(12, 8))
    plot_best_fin(design, show=False)
    png_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  → Guardado: {base_name}")

    return json_path, png_path

def optimize_fins(max_time=60000000, use_steady_state=True):
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
    use_steady_state : bool
        Si True, usa el solver estacionario (RÁPIDO).
        Si False, usa el solver transitorio (LENTO).
    """

    start_time = time.time()
    best = None
    time_exceeded = False
    optimal_iteration = 0  # Contador de diseños óptimos encontrados

    # Indicar qué solver se está usando
    solver_type = "ESTACIONARIO (rápido)" if use_steady_state else "TRANSITORIO (lento)"
    print(f"\n{'='*60}")
    print(f"OPTIMIZACIÓN DE ALETAS - Solver: {solver_type}")
    print(f"{'='*60}\n")

    # BÚSQUEDA EXPANDIDA - Más valores para explorar mejor el espacio de diseño
    # Formato: np.linspace(min, max, steps)
    thickness_values = np.linspace(0.000005, 0.00001, 15)  # 0.005mm - 0.5mm (15 valores)
    height_values    = np.linspace(0.015, 0.025, 10)      # 15mm - 30mm (10 valores)

    # Forma: parámetro 1 → angostura de la punta
    tip_ratios       = np.linspace(0.05, 1.0, 10)  # 0.05 muy angosta hasta 1.0 rectangular (10 valores)

    # Forma: parámetro 2 → redondez
    round_factors    = np.linspace(0.0, 1.0, 8)  # 0 = rectos, 1.0 = muy redondeada (8 valores)

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
                        round_factor=rf,
                        check_balance=False,  # Desactivar heat balance para optimización más rápida
                        use_steady_state=use_steady_state  # Usar solver especificado
                    )

                    if (res is None) or (not res["feasible"]):
                        print("  → no factible")
                        continue

                    print(f"  → factible, masa total = {res['m_total']:.4f} kg")

                    if (best is None) or (res["m_total"] < best["m_total"]):
                        best = res
                        optimal_iteration += 1
                        print("  *** nuevo mejor diseño ***")

                        # Guardar diseño óptimo como PNG y JSON
                        save_optimal_design(best, optimal_iteration)

                if time_exceeded:
                    break
            if time_exceeded:
                break
        if time_exceeded:
            break

    elapsed_time = time.time() - start_time

    # Calcular total de combinaciones evaluadas
    total_combinations = len(thickness_values) * len(height_values) * len(tip_ratios) * len(round_factors)

    print(f"\n{'='*60}")
    print(f"Optimización completada:")
    print(f"  - Solver usado: {solver_type}")
    print(f"  - Tiempo transcurrido: {elapsed_time:.2f} segundos")
    print(f"  - Diseños óptimos encontrados: {optimal_iteration}")
    print(f"  - Combinaciones totales posibles: {total_combinations}")
    print(f"  - Resultados guardados en: optimization_results/")
    print(f"{'='*60}\n")

    return best, elapsed_time, optimal_iteration


if __name__ == "__main__":
    best, elapsed_time, num_optimal = optimize_fins()

    print("\n========================")
    print("  RESUMEN FINAL")
    print("========================")
    print(f"Tiempo de optimización: {elapsed_time:.2f} segundos")
    print(f"Diseños óptimos guardados: {num_optimal}")
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


