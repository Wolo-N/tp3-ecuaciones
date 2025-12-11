"""
Heat Sink Fin Geometry Module

Defines single fin geometry with variable thickness for FVM thermal analysis.
The 2D grid represents a cross-section of ONE fin, extruded by fin_depth.

Coordinate system:
- X: horizontal (fin width/thickness direction)
- Y: vertical (fin height direction)
- Z: depth into page (fin_depth, fixed at 50mm)
"""

import numpy as np

# Import grid data from existing module
from grid_matrices_2d_with_centers import (
    structural_coords,
    center_nodes,
    neighbours,
    areas,
    nHeight,
    nWidth,
    blocks
)

# =============================================================================
# GEOMETRY PARAMETERS
# =============================================================================

# Number of control volumes
n_centers = len(center_nodes)

# Fin depth (extrusion into page) - fixed by processor width
fin_depth = 0.05  # [m] = 50 mm

# Variable thickness per control volume (optimization variable)
# Initially uniform at 2mm
thickness = np.ones(n_centers) * 0.002  # [m] = 2 mm per CV

# Control volumes for thermal mass calculation
# Volume = Area (from 2D grid) × fin_depth
volumes = areas * fin_depth  # [m³]

# =============================================================================
# MATERIAL PROPERTIES (Aluminum)
# =============================================================================

rho = 2700.0    # Density [kg/m³]
k = 205.0       # Thermal conductivity [W/m·K]
cp = 900.0      # Specific heat [J/kg·K]

# =============================================================================
# HEAT SINK PARAMETERS
# =============================================================================

N_fins = 10         # Number of fins in the heat sink
Q_total = 500.0     # Total heat input [W]
h_conv = 300.0      # Convection coefficient [W/m²·K]
T_amb = 45.0        # Ambient temperature [°C]
T_max = 90.0        # Maximum allowable temperature [°C]
max_height = 0.025  # Maximum fin height [m] = 25 mm

# =============================================================================
# BOUNDARY IDENTIFICATION
# =============================================================================

def identify_boundaries():
    """
    Identify boundary nodes based on their neighbor connectivity.

    Returns
    -------
    boundaries : dict
        Dictionary with keys:
        - 'base': indices of bottom row CVs (heat input)
        - 'tip': indices of top row CVs
        - 'left': indices of left edge CVs
        - 'right': indices of right edge CVs
        - 'all_exposed': all CVs with convection surfaces
    """
    base_nodes = []      # Bottom row (no down neighbor)
    tip_nodes = []       # Top row (no up neighbor)
    left_nodes = []      # Left edge (no left neighbor)
    right_nodes = []     # Right edge (no right neighbor)

    for i, center in enumerate(center_nodes):
        nb = neighbours[i]  # [Center, Left, Right, Up, Down]

        # Check each direction
        if nb[1] is None:  # No left neighbor
            left_nodes.append(i)
        if nb[2] is None:  # No right neighbor
            right_nodes.append(i)
        if nb[3] is None:  # No up neighbor
            tip_nodes.append(i)
        if nb[4] is None:  # No down neighbor
            base_nodes.append(i)

    return {
        'base': np.array(base_nodes),
        'tip': np.array(tip_nodes),
        'left': np.array(left_nodes),
        'right': np.array(right_nodes),
        'all_exposed': np.unique(np.concatenate([base_nodes, tip_nodes, left_nodes, right_nodes]))
    }

boundaries = identify_boundaries()

# =============================================================================
# GEOMETRY FUNCTIONS
# =============================================================================

def compute_mass(thickness_array=None):
    """
    Compute total mass of a single fin.

    Parameters
    ----------
    thickness_array : ndarray, optional
        Thickness per CV [m]. If None, uses global thickness.

    Returns
    -------
    mass : float
        Total fin mass [kg]
    """
    if thickness_array is None:
        thickness_array = thickness

    # Mass = rho * sum(area * thickness * fin_depth)
    # Note: Each CV contributes area * thickness (cross-section) * fin_depth
    mass = rho * np.sum(areas * thickness_array * fin_depth)
    return mass


def compute_total_heatsink_mass(thickness_array=None):
    """
    Compute total mass of entire heat sink (all fins).

    Parameters
    ----------
    thickness_array : ndarray, optional
        Thickness per CV [m]. If None, uses global thickness.

    Returns
    -------
    total_mass : float
        Total heat sink mass [kg]
    """
    return compute_mass(thickness_array) * N_fins


def get_cv_coordinates():
    """
    Get the (x, y) coordinates of each control volume center.

    Returns
    -------
    coords : ndarray (n_centers, 2)
        Coordinates of CV centers
    """
    return structural_coords[center_nodes]


def compute_neighbor_distances():
    """
    Compute distances between neighboring control volumes.

    Returns
    -------
    distances : ndarray (n_centers, 4)
        Distance to [Left, Right, Up, Down] neighbors.
        NaN if no neighbor exists.
    """
    coords = get_cv_coordinates()
    distances = np.full((n_centers, 4), np.nan)

    for i in range(n_centers):
        nb = neighbours[i]  # [Center, Left, Right, Up, Down]
        center_coord = coords[i]

        # Map global indices to local CV indices
        global_to_local = {node: idx for idx, node in enumerate(center_nodes)}

        for j, direction in enumerate([1, 2, 3, 4]):  # L, R, U, D
            if nb[direction] is not None:
                neighbor_local = global_to_local.get(nb[direction])
                if neighbor_local is not None:
                    neighbor_coord = coords[neighbor_local]
                    distances[i, j] = np.linalg.norm(neighbor_coord - center_coord)

    return distances


def compute_conduction_areas(thickness_array=None):
    """
    Compute conduction areas between neighboring CVs.

    Conduction area depends on:
    - thickness at the interface (average of both CVs)
    - fin_depth (fixed)

    Parameters
    ----------
    thickness_array : ndarray, optional
        Thickness per CV [m]. If None, uses global thickness.

    Returns
    -------
    cond_areas : ndarray (n_centers, 4)
        Conduction area to [Left, Right, Up, Down] neighbors [m²].
        0 if no neighbor exists.
    """
    if thickness_array is None:
        thickness_array = thickness

    cond_areas = np.zeros((n_centers, 4))

    # Map global indices to local CV indices
    global_to_local = {node: idx for idx, node in enumerate(center_nodes)}

    for i in range(n_centers):
        nb = neighbours[i]  # [Center, Left, Right, Up, Down]

        for j, direction in enumerate([1, 2, 3, 4]):  # L, R, U, D
            if nb[direction] is not None:
                neighbor_local = global_to_local.get(nb[direction])
                if neighbor_local is not None:
                    # Average thickness at interface
                    avg_thickness = 0.5 * (thickness_array[i] + thickness_array[neighbor_local])
                    # Conduction area = thickness * fin_depth
                    cond_areas[i, j] = avg_thickness * fin_depth

    return cond_areas


def compute_convection_areas(thickness_array=None):
    """
    Compute convection areas for each CV.

    Convection occurs on exposed surfaces:
    - Front and back faces (2 * area of CV)
    - Edge faces where no neighbor exists (thickness * edge_length)

    Parameters
    ----------
    thickness_array : ndarray, optional
        Thickness per CV [m]. If None, uses global thickness.

    Returns
    -------
    conv_areas : ndarray (n_centers,)
        Total convection area per CV [m²]
    """
    if thickness_array is None:
        thickness_array = thickness

    conv_areas = np.zeros(n_centers)
    distances = compute_neighbor_distances()

    for i in range(n_centers):
        nb = neighbours[i]  # [Center, Left, Right, Up, Down]

        # Front and back faces (2D area * 2)
        # Actually for a fin cross-section, convection is on the perimeter
        # The "area" from the grid is the 2D cross-section area
        # Convection surface = perimeter * fin_depth + edges without neighbors

        # For simplicity, assume convection on surfaces without neighbors
        edge_area = 0.0

        for j, direction in enumerate([1, 2, 3, 4]):  # L, R, U, D
            if nb[direction] is None:
                # Exposed edge - estimate edge length from grid spacing
                # Use average distance to existing neighbors as estimate
                valid_distances = distances[i, ~np.isnan(distances[i, :])]
                if len(valid_distances) > 0:
                    edge_length = np.mean(valid_distances)
                else:
                    edge_length = 0.002  # Default 2mm

                # Edge convection area = edge_length * fin_depth
                edge_area += edge_length * fin_depth

        # Front/back surfaces (the 2D cross-section viewed from depth direction)
        # These are always exposed for convection
        front_back_area = 2 * areas[i]  # 2 faces

        conv_areas[i] = edge_area + front_back_area

    return conv_areas


# =============================================================================
# SUMMARY OUTPUT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HEAT SINK FIN GEOMETRY")
    print("=" * 70)

    print(f"\nGrid Information:")
    print(f"  - Total CVs: {n_centers}")
    print(f"  - Grid size: {nHeight} x {nWidth} nodes")

    print(f"\nGeometry Parameters:")
    print(f"  - Fin depth: {fin_depth * 1000:.1f} mm")
    print(f"  - Initial thickness: {thickness[0] * 1000:.1f} mm (uniform)")
    print(f"  - Max fin height: {max_height * 1000:.1f} mm")

    print(f"\nMaterial Properties (Aluminum):")
    print(f"  - Density: {rho} kg/m³")
    print(f"  - Thermal conductivity: {k} W/m·K")
    print(f"  - Specific heat: {cp} J/kg·K")

    print(f"\nHeat Sink Parameters:")
    print(f"  - Number of fins: {N_fins}")
    print(f"  - Total heat input: {Q_total} W")
    print(f"  - Heat per fin: {Q_total/N_fins:.1f} W")
    print(f"  - Convection coefficient: {h_conv} W/m²·K")
    print(f"  - Ambient temperature: {T_amb}°C")
    print(f"  - Max temperature constraint: {T_max}°C")

    print(f"\nBoundary Nodes:")
    print(f"  - Base (heat input): {len(boundaries['base'])} nodes")
    print(f"  - Tip: {len(boundaries['tip'])} nodes")
    print(f"  - Left edge: {len(boundaries['left'])} nodes")
    print(f"  - Right edge: {len(boundaries['right'])} nodes")

    print(f"\nMass Calculation:")
    single_fin_mass = compute_mass()
    total_mass = compute_total_heatsink_mass()
    print(f"  - Single fin mass: {single_fin_mass * 1000:.2f} g")
    print(f"  - Total heat sink mass: {total_mass * 1000:.2f} g")

    print("=" * 70)
