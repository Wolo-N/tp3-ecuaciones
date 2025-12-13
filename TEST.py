from optimize_fins import plot_temperature_heatmap, plot_temperature_3d

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

print('Generando heatmap de temperaturas para diseño de prueba...')
plot_temperature_heatmap(test_design)

print('\nGenerando visualización 3D con disipación en Z...')
#plot_temperature_3d(test_design)
