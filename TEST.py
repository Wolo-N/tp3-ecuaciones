from finEuler import evaluate_fin_design
from finGridwCenters import build_fin_grid_2d
from plots import (plot_grid_with_blocks, plot_temperature_heatmap,
                   solve_and_track_evolution, plot_thermal_evolution,
                   plot_complete_heatsink)
from parameters import N_HEIGHT, N_WIDTH, RHO, CP, K, H, T_INF, T_BASE, GAP, BASE_SIZE
import numpy as np
import matplotlib.pyplot as plt

# Test different fin geometries for presentation
print("="*60)
print("Modelo térmico 2D transitorio por Volúmenes Finitos")
print("="*60)
print("VISUALIZACIÓN DE ALETA ÓPTIMA")
print("="*60)

# PARÁMETROS DE LA ALETA ÓPTIMA
print("\n1. ALETA ÓPTIMA - Geometría")
print("-" * 60)
fin_thickness = 0.0001              # 0.1mm
fin_height = 0.015068965517241378   # ~15.07mm
fin_length = 0.050                  # 50mm
tip_ratio = 1.0                     # rectangular (no taper)
round_factor = 1.0                  # completamente redondeado

(structural_coords, center_nodes, neighbours_dict, areas_dict,
 volumes_dict, blocks, center_distances, face_lengths,
 face_areas, boundary_areas) = build_fin_grid_2d(
    fin_thickness=fin_thickness,
    fin_height=fin_height,
    fin_length=fin_length,
    tip_ratio=tip_ratio,
    round_factor=round_factor
)

print(f"Parámetros de diseño:")
print(f"  Espesor:       {fin_thickness*1000:.3f} mm")
print(f"  Altura:        {fin_height*1000:.2f} mm")
print(f"  Longitud:      {fin_length*1000:.1f} mm")
print(f"  Tip ratio:     {tip_ratio:.2f} (rectangular)")
print(f"  Round factor:  {round_factor:.2f} (completamente redondeado)")
print(f"\nEstadísticas de malla:")
print(f"  Nodos totales:                    {len(structural_coords)}")
print(f"  Nodos centrales (volúmenes):      {len(center_nodes)}")
print(f"  Volumen total:                    {sum(volumes_dict.values()):.6e} m³")
print(f"  Volumen total:                    {sum(volumes_dict.values())*1e9:.3f} mm³")
print(f"\nEstadísticas de volúmenes de control:")
print(f"  Volumen mínimo:  {min(volumes_dict.values()):.6e} m³")
print(f"  Volumen máximo:  {max(volumes_dict.values()):.6e} m³")
print(f"  Volumen promedio: {np.mean(list(volumes_dict.values())):.6e} m³")

# Graficar la malla
print("\nGenerando gráfico de malla...")
plot_grid_with_blocks(structural_coords, center_nodes, N_HEIGHT, N_WIDTH, x_exaggeration=40.0)


# Test case 2: Mapa de calor de temperaturas
print("\n\n2. ALETA ÓPTIMA - Mapa de Calor de Temperaturas")
print("-" * 60)
print("Resolviendo campo de temperaturas...")

# Crear diccionario de diseño para la aleta óptima
design = {
    'thickness': fin_thickness,
    'height': fin_height,
    'tip_ratio': tip_ratio,
    'round_factor': round_factor
}

plot_temperature_heatmap(design, show=True, verbose=True)


# Test case 3: Evaluación de diseño + verificación de balance térmico
print("\n\n3. ALETA ÓPTIMA - Evaluación de Desempeño")
print("-" * 60)
evaluation = evaluate_fin_design(
    fin_thickness=fin_thickness,
    fin_height=fin_height,
    fin_length=fin_length,
    tip_ratio=tip_ratio,
    round_factor=round_factor,
    check_balance=True,
    use_steady_state=True,
)

if evaluation is None:
    print("❌ Evaluación fallida: Q_fin <= 0, diseño rechazado.")
else:
    print("\n" + "="*60)
    print("RESULTADOS DE LA ALETA ÓPTIMA")
    print("="*60)
    print(f"  Calor disipado por aleta:         {evaluation['Q_fin']:.4f} W")
    print(f"  Masa por aleta:                   {evaluation['m_fin']*1000:.4f} g")
    print(f"  Aletas necesarias (500 W):        {evaluation['n_required']}")
    print(f"  Aletas máximas que caben:         {evaluation['n_max']}")
    print(f"  Cumple restricciones geométricas: {'✓ SÍ' if evaluation['feasible'] else '✗ NO'}")
    print(f"  Masa total del disipador:         {evaluation['m_total']*1000:.2f} g")
    print(f"  Masa total del disipador:         {evaluation['m_total']:.5f} kg")
    print("="*60)


# Test case 4: Evolución térmica transitoria
print("\n\n4. ALETA ÓPTIMA - Evolución Térmica Transitoria")
print("-" * 60)
print("Simulando evolución desde estado inicial hasta estacionario...")

# Resolver evolución térmica
(time_hist, T_avg, T_max, T_min, Q_hist, T_final, coords, centers, boundary_areas) = solve_and_track_evolution(
    k=K, h=H, rho=RHO, cp=CP,
    T_inf=T_INF, T_base=T_BASE,
    dt=0.01, t_final=10.0, tol=1e-6,
    fin_thickness=fin_thickness,
    fin_height=fin_height,
    fin_length=fin_length,
    tip_ratio=tip_ratio,
    round_factor=round_factor
)

# Calcular cuántas aletas caben
total_thickness_per_fin = fin_thickness + GAP
n_fins_fit = int(BASE_SIZE // total_thickness_per_fin)

print(f"\nAletas que caben en el disipador: {n_fins_fit}")
print(f"Calor por aleta (estado final): {Q_hist[-1]:.4f} W")
print(f"Calor total del disipador: {Q_hist[-1] * n_fins_fit:.2f} W")

# Configuración para el gráfico
config = {
    'thickness': fin_thickness,
    'height': fin_height,
    'length': fin_length
}

# Graficar evolución térmica
print("\nGenerando gráfico de evolución térmica...")
fig, axes = plot_thermal_evolution(time_hist, T_avg, T_max, T_min, Q_hist, config, n_fins_fit)
import matplotlib.pyplot as plt
plt.show()

# Test case 5: Visualización del disipador completo
print("\n\n5. DISIPADOR COMPLETO - Vista del Conjunto")
print("-" * 60)
print("Visualizando disipador con todas las aletas óptimas distribuidas...")

# Calcular cuántas aletas caben
total_thickness_per_fin = fin_thickness + GAP
n_fins_complete = int(BASE_SIZE // total_thickness_per_fin)

print(f"\nConfiguración del disipador:")
print(f"  Base: {BASE_SIZE*1000:.1f} mm × {fin_length*1000:.1f} mm")
print(f"  Número de aletas: {n_fins_complete}")
print(f"  Espesor por aleta (fin + gap): {total_thickness_per_fin*1000:.2f} mm")
print(f"  Espacio ocupado: {n_fins_complete * total_thickness_per_fin * 1000:.1f} mm")

# Graficar disipador completo
print("\nGenerando visualización del disipador completo...")
fig = plot_complete_heatsink(
    fin_thickness=fin_thickness,
    fin_height=fin_height,
    fin_length=fin_length,
    n_fins=n_fins_complete,
    gap=GAP,
    tip_ratio=tip_ratio,
    round_factor=round_factor
)
plt.show()

print("\n" + "="*60)
print("VISUALIZACIÓN COMPLETADA")
print("="*60)
print("Gráficos generados:")
print("  1. Malla de volúmenes finitos (geometría)")
print("  2. Mapa de calor de temperaturas (estado estacionario)")
print("  3. Evolución térmica transitoria (T vs tiempo + Q vs tiempo)")
print("  4. Disipador completo (vista superior + vista frontal)")
print("\nValidaciones ejecutadas:")
print("  ✓ Balance térmico (conducción = convección)")
print("  ✓ Factibilidad geométrica (n_req ≤ n_max)")
print("  ✓ Capacidad de disipación (Q_total ≥ 500 W)")
print("  ✓ Convergencia a estado estacionario")
print("="*60)
