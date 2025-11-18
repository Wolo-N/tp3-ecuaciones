"""
Discretized volume equation, forward Euler in time

**Governing equation**:
$$ k\,A_W\,\frac{T_W - T_b}{dx} + h_b \Delta x \,P_b\,\big(T_{\infty} - T_b \big) + k\, A_E\,\frac{T_E -T_b }{\Delta x}= \rho A(x) C \frac{T_b^{Next} - T_b}{\Delta t}$$

**Features**
- Variable **geometry** via user-defined `A(x)` and `P(x)` (default: tapered circular pin fin).
- Variable **material** and **convection**: `k(x)`, `h(x)`.
- Base boundary at `x=0`: **Dirichlet** temperature (easy to switch to heat flux).
- Tip at `x=L`: **convective (Robin)** or **insulated**.
- Computes temperature field, base heat rate, and integrated convective losses.
- Clean functions for reuse.
"""

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt

"""## Parameters"""

# Fin
lFin = 0.01          # fin height [m]
nVolumes = 10           # number of control volumes (nodes)

# Boundary Conditions
T_inf = 45.0 #[C]
T_base = 60.0 #[C]

# Material properties
k = 205.0          # [W/m-K]
h = 30.0           # [W/m^2-K]
c = 902.0 #[J/kg/C]

# Elliptic Fin
R0 = 0.0025    # base major radius [m]
r0 = 0.001    # base minradius [m]
RL = 0.001    # tip major [m]
rL = 0.001    # tip minor radius [m]

def r_fun(x):
    return (r0 + (rL - r0) * (x / lFin), R0 + (RL - R0) * (x / lFin))

def A_fun(x):
    (R, r) = r_fun(x)
    return pi * R * r

def P_fun(x):
    (R, r) = r_fun(x)
    return 2.0 * sqrt((R**2+r**2)/2)

"""## Finite Volume Solver"""

def build_grid(lFin, nVolumes):
    x = np.linspace(0.0, lFin, nVolumes)
    Ax=x[1]-x[0]

    # Neighbours
    neighbours = np.empty((nVolumes, 3), dtype=object)
    neighbours[0, :] = [None, 0, 1]                               # first volumen
    neighbours[nVolumes-1, :] = [nVolumes-2, nVolumes-1, None]    # last volumen
    for i in range(1, nVolumes-1):                                # internal volumes
        neighbours[i, :] = [i-1, i, i+1]
    # Areas
    areas = np.empty((nVolumes+1, 3)) # [m2] Area W, Area E, Perimeter
    areas[0, :] = [A_fun(0), A_fun(Ax), P_fun(Ax/2)*Ax]       # first volumen areas
    areas[nVolumes, :] = [A_fun(lFin-Ax), A_fun(lFin), P_fun(lFin-Ax/2)*Ax]        # last volumen areas
    for i in range(1, nVolumes):                            # internal volumes areas
        areas[i, :] = [A_fun(i*Ax), A_fun((i+1)*Ax), P_fun((i+0.5)*Ax)*Ax]
    return x, areas, neighbours,Ax

def assemble_system(areas,neighbours,nVolumes):
    # System matrices
    M = np.zeros((nVolumes,nVolumes), dtype=float)
    V = np.zeros(nVolumes, dtype=float)

    # Interior nodes
    for i in range (1,len(neighbours[:,1])-1):
        M[i,neighbours[i,0]] = k / Ax * areas[i,0]
        M[i,neighbours[i,2]] = k / Ax * areas[i,1]
        M[i,neighbours[i,1]] = k / Ax * (-areas[i,0]-areas[i,1]) - h * areas[i,2] * Ax
        V[i]                 = - h * areas[i,2] * Ax * T_inf

    # Left boundary fixed Temperature
    M[0, 0] = 1.0
    V[0] = T_base

    # Right boundary
    M[nVolumes-1,neighbours[nVolumes-1,0]] = k / Ax * areas[nVolumes-1,0]
    M[nVolumes-1,neighbours[nVolumes-1,1]] = k / Ax * (-areas[nVolumes-1,0]) - h * (areas[nVolumes-1,1] + areas[nVolumes-1,2] * Ax/2)
    V[nVolumes-1]                          = - h * (areas[nVolumes-1,1] + areas[nVolumes-1,2] * Ax/2)  * T_inf

    return M,V

"""## Solve and Plot"""

x, areas, neighbours, Ax = build_grid(lFin, nVolumes)

print(neighbours)

print(areas)

M,V = assemble_system(areas,neighbours,nVolumes)

# Solve
T = np.linalg.solve(M, V)

# Calculate heat convected at x=0
T0 = T[0]  # Temperature at x=0
T1 = T[1]  # Temperature at first interior node
A0_boundary = areas[0, 1]  # Area between node 0 and node 1 (Area E of node 0)
P0 = P_fun(Ax/2)  # Perimeter at x=0 (using midpoint of first control volume)

Q_convected = k * A0_boundary * (T1 - T0) / Ax + h * (Ax / 2) * P0 * (T_inf - T0)
#0 = k * A0_boundary * (T1 - T0) / Ax + h * (Ax / 2) * P0 * (T_inf - T0) + Q_convected
#el Q_convected es positivo para contraarrestar las dos otras partes que son las que salen de la aleta

print(f"\nHeat entering fin at x=0: Q = {-Q_convected:.4f} W")
print(f"  - Conduction into fin (to x=dx): {k * A0_boundary * (T0 - T1) / Ax:.4f} W")
print(f"  - Convection from surface (first half-element): {-h * (Ax / 2) * P0 * (T_inf - T0):.4f} W")

#plt.figure(figsize=(8,4.5))
#plt.plot(x, T)
#plt.xlabel("x [m]")
#plt.ylabel("Temperature [°C]")
#plt.title("1D Variable-Area Fin (FVM)")
#plt.grid(True)
#plt.show()