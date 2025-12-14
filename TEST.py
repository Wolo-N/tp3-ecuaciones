from plot_fins import plot_temperature_heatmap, plot_temperature_3d
from fin_2d_forward_euler import evaluate_fin_design

# Test design
test_design = {
    'thickness': 0.01,
    'height': 0.025,
    'tip_ratio': 1.0,
    'round_factor': 0.0,
    'm_total': 0.0948,
    'Q_fin': 14.5,
    'm_fin': 0.0027,
    'n_required': 35,
    'n_max': 49
}

# Evaluate the design
print('Evaluando diseño de aleta...')
result = evaluate_fin_design(
    fin_thickness=test_design['thickness'],
    fin_height=test_design['height'],
    fin_length=0.05,
    gap=0.001,
    tip_ratio=test_design['tip_ratio'],
    round_factor=test_design['round_factor']
)

if result and result['feasible']:
    print(f"\nDiseño factible:")
    print(f"  Q por aleta:       {result['Q_fin']:.3f} W")
    print(f"  Masa por aleta:    {result['m_fin']:.6f} kg")
    print(f"  Aletas requeridas: {result['n_required']:.1f}")
    print(f"  Aletas máximas:    {result['n_max']}")
    print(f"  Masa total:        {result['m_total']:.4f} kg")
else:
    print("\nDiseño no factible")

print('\nGenerando heatmap de temperaturas para diseño de prueba...')
plot_temperature_heatmap(test_design)

print('\nGenerando visualización 3D con disipación en Z...')
#plot_temperature_3d(test_design)
