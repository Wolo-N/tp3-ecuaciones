# optimize_fins.py

from finEuler import evaluate_fin_design
from plots import plot_best_fin
from parameters import GAP
import time
import json
import os
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar sin mostrar
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import math

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


def save_sweep_results(results, best_design=None, output_dir="optimization_results"):
    """
    Guarda todas las combinaciones evaluadas y genera un gráfico
    de efectividad vs espesor/altura.

    Parameters
    ----------
    results : list of dict
        Lista con los resultados de cada combinación probada.
    best_design : dict or None
        Diseño óptimo final para resaltar en el gráfico (opcional).
    output_dir : str
        Directorio donde guardar archivos.

    Returns
    -------
    tuple
        (json_path, scatter_path, lines_path) si existen, en caso contrario (None, None, None)
    """
    if not results:
        return None, None, None

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"sweep_{timestamp}.json")

    # Guardar resultados completos
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Preparar datos factibles para gráfica
    feasible = [r for r in results if r.get("effectiveness") is not None]
    if not feasible:
        return json_path, None, None

    thickness_mm = np.array([r["thickness"] * 1000 for r in feasible])
    height_mm = np.array([r["height"] * 1000 for r in feasible])
    effectiveness = np.array([r["effectiveness"] for r in feasible])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        thickness_mm,
        height_mm,
        c=effectiveness,
        cmap="viridis",
        s=35,
        edgecolors="none"
    )
    ax.set_xlabel("Thickness [mm]")
    ax.set_ylabel("Height [mm]")
    ax.set_title("Fin effectiveness (Q_fin / m_fin) vs geometry")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Effectiveness [W/kg]")

    plot_path = os.path.join(output_dir, f"sweep_effectiveness_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)

    # Línea comparativa masa total vs Q_fin con ejes para thickness/height
    line_path = None
    feasible_sorted = sorted(feasible, key=lambda r: (r["thickness"], r["height"]))
    if feasible_sorted:
        idx = np.arange(len(feasible_sorted))
        total_mass = np.array([r["m_total"] for r in feasible_sorted])
        q_fin = np.array([r["Q_fin"] for r in feasible_sorted])
        thickness_mm = np.array([r["thickness"] * 1000 for r in feasible_sorted])
        height_mm = np.array([r["height"] * 1000 for r in feasible_sorted])

        fig2, ax_q = plt.subplots(figsize=(9, 5))
        ax_m = ax_q.twinx()

        line_q, = ax_q.plot(idx, q_fin, label="Q_fin [W]", color="tab:orange", linewidth=2)
        line_m, = ax_m.plot(idx, total_mass, label="Total mass [kg]", color="tab:blue", linewidth=2)

        ax_q.set_xlabel("Thickness [mm]")
        ax_q.set_ylabel("Q_fin [W]", color="tab:orange")
        ax_m.set_ylabel("Total mass [kg]", color="tab:blue")
        ax_q.set_title("Total mass and Q_fin across evaluated designs")
        ax_q.grid(True, linestyle="--", alpha=0.4)
        ax_q.tick_params(axis="y", labelcolor="tab:orange")
        ax_m.tick_params(axis="y", labelcolor="tab:blue")

        lines = [line_q, line_m]
        labels = [line.get_label() for line in lines]

        # Configurar ticks limitados para legibilidad
        max_ticks = 10
        step = max(1, len(idx) // max_ticks)
        tick_positions = list(idx[::step])
        if tick_positions[-1] != idx[-1]:
            tick_positions.append(idx[-1])
        tick_labels_thickness = [f"{thickness_mm[i]:.3f}" for i in tick_positions]
        ax_q.set_xticks(tick_positions)
        ax_q.set_xticklabels(tick_labels_thickness, rotation=45, ha="right")

        # Segundo eje X mostrando alturas correspondientes
        ax_top = ax_q.twiny()
        ax_top.set_xlim(ax_q.get_xlim())
        ax_top.set_xticks(tick_positions)
        tick_labels_height = [f"{height_mm[i]:.2f}" for i in tick_positions]
        ax_top.set_xticklabels(tick_labels_height, rotation=45, ha="left")
        ax_q.set_xlabel("Thickness [mm]")
        ax_top.set_xlabel("Height [mm]")

        best_idx = None
        if best_design is not None:
            for i, r in enumerate(feasible_sorted):
                if (math.isclose(r["thickness"], best_design["thickness"], rel_tol=1e-9, abs_tol=1e-12) and
                        math.isclose(r["height"], best_design["height"], rel_tol=1e-9, abs_tol=1e-12)):
                    best_idx = i
                    break

        if best_idx is not None:
            scatter_best = ax_q.scatter(
                best_idx,
                q_fin[best_idx],
                marker="x",
                s=100,
                c="red",
                linewidths=2,
                label="Mejor diseño"
            )
            ax_m.scatter(
                best_idx,
                total_mass[best_idx],
                marker="x",
                s=100,
                c="red",
                linewidths=2
            )
            lines.append(scatter_best)
            labels.append("Mejor diseño")

        ax_q.legend(lines, labels, loc="upper right")

        line_path = os.path.join(output_dir, f"sweep_mass_q_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(line_path, dpi=200)
        plt.close(fig2)

    return json_path, plot_path, line_path

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

    Returns
    -------
    tuple
        (best_design, elapsed_time, optimal_iteration)
        Además se generan:
        - sweep_<timestamp>.json con todas las combinaciones evaluadas
        - sweep_effectiveness_<timestamp>.png con el mapa de efectividad
    """

    start_time = time.time()
    best = None
    time_exceeded = False
    optimal_iteration = 0  # Contador de diseños óptimos encontrados
    all_results = []  # Almacena todas las combinaciones (factibles o no)

    # Indicar qué solver se está usando
    solver_type = "ESTACIONARIO (rápido)" if use_steady_state else "TRANSITORIO (lento)"
    print(f"\n{'='*60}")
    print(f"OPTIMIZACIÓN DE ALETAS - Solver: {solver_type}")
    print(f"{'='*60}\n")

    # Formato: np.linspace(min, max, steps)
    thickness_values = np.linspace(0.0001, 0.001, 30)  # 0.005mm - 0.1mm (15 valores)
    height_values    = np.linspace(0.001, 0.025, 30)      # 15mm - 30mm (10 valores)

    # Forma: parámetro 1 → angostura de la punta
    #tip_ratios       = np.linspace(0.05, 1.0, 10)  # 0.05 muy angosta hasta 1.0 rectangular (10 valores)
    tip_ratios       = [1]

    # Forma: parámetro 2 → redondez
    #round_factors    = np.linspace(0.0, 1.0, 8)  # 0 = rectos, 1.0 = muy redondeada (8 valores)
    round_factors    = [1]
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
                        gap=GAP,  # Espacio entre aletas desde parameters.py
                        tip_ratio=tr,
                        round_factor=rf,
                        check_balance=False,  # Desactivar heat balance para optimización más rápida
                        use_steady_state=use_steady_state  # Usar solver especificado
                    )

                    result_entry = {
                        "thickness": float(t),
                        "height": float(h),
                        "tip_ratio": float(tr),
                        "round_factor": float(rf),
                        "feasible": False,
                        "Q_fin": None,
                        "m_fin": None,
                        "n_required": None,
                        "n_max": None,
                        "m_total": None,
                        "effectiveness": None  # Definido como Q_fin / m_fin
                    }

                    if (res is None) or (not res["feasible"]):
                        print("  → no factible")
                        all_results.append(result_entry)
                        continue

                    print(f"  → factible, masa total = {res['m_total']:.4f} kg")

                    effectiveness = None
                    if res["m_fin"] > 0:
                        effectiveness = res["Q_fin"] / res["m_fin"]

                    result_entry.update({
                        "feasible": True,
                        "Q_fin": float(res["Q_fin"]),
                        "m_fin": float(res["m_fin"]),
                        "n_required": int(res["n_required"]),
                        "n_max": int(res["n_max"]),
                        "m_total": float(res["m_total"]),
                        "effectiveness": float(effectiveness) if effectiveness is not None else None
                    })
                    all_results.append(result_entry)

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

    sweep_json_path, sweep_plot_path, sweep_line_path = save_sweep_results(all_results, best_design=best)

    print(f"\n{'='*60}")
    print(f"Optimización completada:")
    print(f"  - Solver usado: {solver_type}")
    print(f"  - Tiempo transcurrido: {elapsed_time:.2f} segundos")
    print(f"  - Diseños óptimos encontrados: {optimal_iteration}")
    print(f"  - Combinaciones totales posibles: {total_combinations}")
    if sweep_json_path:
        print(f"  - Sweep JSON: {sweep_json_path}")
    if sweep_plot_path:
        print(f"  - Gráfico efectividad: {sweep_plot_path}")
    if sweep_line_path:
        print(f"  - Línea masa/Q_fin: {sweep_line_path}")
    if not sweep_plot_path and not sweep_line_path:
        print("  - No se generaron gráficos (sin diseños factibles).")
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
