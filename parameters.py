# parameters.py
# Parámetros físicos y de operación del disipador de calor

# Propiedades del material (Aluminio)
RHO = 2700      # Densidad [kg/m³]
CP = 900        # Capacidad calorífica específica [J/kg·K]
K = 205         # Conductividad térmica [W/m·K]

# Condiciones de transferencia de calor
H = 300         # Coeficiente de convección [W/m²·K]
T_INF = 45     # Temperatura ambiente (peor caso) [°C]
T_BASE = 90     # Temperatura de la base (CPU) [°C]

# Parámetros del problema
Q_REQUIRED = 500    # Calor total a disipar [W]
BASE_SIZE = 0.05    # Tamaño de la base del disipador [m] (50 mm)
GAP = 0.001         # Espacio entre aletas consecutivas [m] (1 mm)

# Discretización de la malla FVM
N_HEIGHT = 11   # Número de nodos en dirección Y (altura) - debe ser impar
N_WIDTH = 11    # Número de nodos en dirección X (espesor) - debe ser impar
