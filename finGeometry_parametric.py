"""
Parametric Fin Geometry Generator

Allows optimization of fin geometry by accepting parameters:
- fin_thickness: thickness at base (m)
- fin_height: height of fin (m)
- taper_ratio: ratio of tip_width/base_width (0-1, where 1=rectangular, <1=tapered)

Based on finGeometry.py but with configurable parameters.
"""

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches


def build_parametric_fin_grid(fin_thickness=0.002, fin_height=0.025,
                               taper_ratio=1.0, fin_depth=0.050,
                               nHeight=11, nWidth=11):
    """
    Build 2D FVM grid for a single fin with parametric geometry.

    Parameters
    ----------
    fin_thickness : float
        Fin thickness at base (m)
    fin_height : float
        Fin height (m)
    taper_ratio : float
        Ratio of tip thickness to base thickness (0-1)
        1.0 = rectangular fin
        <1.0 = tapered fin (thinner at top)
    fin_depth : float
        Fin depth/length (extrusion in Z direction) (m)
    nHeight : int
        Number of nodes in height direction
    nWidth : int
        Number of nodes in width direction

    Returns
    -------
    Same as build_fin_grid_2d() from finGeometry.py
    """

    # ---------------- Q9 Shape functions ----------------
    def q9_shape_functions(xi, eta):
        L1 = 0.5 * xi * (xi - 1.0)
        L2 = 1.0 - xi**2
        L3 = 0.5 * xi * (xi + 1.0)

        M1 = 0.5 * eta * (eta - 1.0)
        M2 = 1.0 - eta**2
        M3 = 0.5 * eta * (eta + 1.0)

        N = np.array([
            L1*M1,  # N1 (-1,-1)
            L3*M1,  # N2 ( 1,-1)
            L3*M3,  # N3 ( 1, 1)
            L1*M3,  # N4 (-1, 1)
            L2*M1,  # N5 ( 0,-1)
            L3*M2,  # N6 ( 1, 0)
            L2*M3,  # N7 ( 0, 1)
            L1*M2,  # N8 (-1, 0)
            L2*M2   # N9 ( 0, 0)
        ])
        return N

    def q9_interpolate_points(ctrl_pts, natural_coords):
        ctrl_pts = np.asarray(ctrl_pts, dtype=float).reshape(9, 2)
        natural_coords = np.asarray(natural_coords, dtype=float).reshape(-1, 2)
        Nmat = np.vstack([q9_shape_functions(xi, eta) for xi, eta in natural_coords])
        structural_coords = Nmat @ ctrl_pts
        return structural_coords

    # -------- Create natural coordinate grid --------
    h = np.linspace(-1, 1, nHeight)
    w = np.linspace(-1, 1, nWidth)
    H, W = np.meshgrid(h, w)
    natural_coords = np.column_stack((W.ravel(), H.ravel()))

    # -------- Define control points for tapered fin --------
    half_t_base = fin_thickness / 2.0
    half_t_tip = (fin_thickness * taper_ratio) / 2.0
    half_t_mid = (half_t_base + half_t_tip) / 2.0

    # Q9 control points (tapered or rectangular depending on taper_ratio)
    ctrl = np.array([
        [-half_t_base,      0.0],         # N1: bottom-left
        [ half_t_base,      0.0],         # N2: bottom-right
        [ half_t_tip,  fin_height],       # N3: top-right
        [-half_t_tip,  fin_height],       # N4: top-left
        [ 0.0,              0.0],         # N5: bottom-center
        [ half_t_mid,  fin_height/2.0],   # N6: mid-right
        [ 0.0,         fin_height],       # N7: top-center
        [-half_t_mid,  fin_height/2.0],   # N8: mid-left
        [ 0.0,         fin_height/2.0]    # N9: center
    ])

    # -------- Coordinate transformation --------
    structural_coords = q9_interpolate_points(ctrl, natural_coords)

    # -------- Select center nodes --------
    def get_center_nodes(nHeight, nWidth):
        center_nodes = []
        center_nodes_grid = {}
        for row in range(0, nHeight, 2):
            for col in range(0, nWidth, 2):
                node_index = row * nWidth + col
                center_nodes.append(node_index)
                center_nodes_grid[(row, col)] = node_index
        return center_nodes, center_nodes_grid

    center_nodes, center_nodes_grid = get_center_nodes(nHeight, nWidth)

    # -------- Define blocks --------
    def get_block_nodes(center_idx, nWidth, nHeight):
        center_row = center_idx // nWidth
        center_col = center_idx % nWidth
        block = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r = center_row + dr
                c = center_col + dc
                if 0 <= r < nHeight and 0 <= c < nWidth:
                    block.append(r * nWidth + c)
        return block

    blocks = {center: get_block_nodes(center, nWidth, nHeight) for center in center_nodes}

    # -------- Neighbor connectivity --------
    def get_center_neighbors_from_grid(center_idx, center_nodes_grid, nWidth):
        center_row = center_idx // nWidth
        center_col = center_idx % nWidth
        neighbors = [center_idx, None, None, None, None]

        left_key = (center_row, center_col - 2)
        if left_key in center_nodes_grid:
            neighbors[1] = center_nodes_grid[left_key]

        right_key = (center_row, center_col + 2)
        if right_key in center_nodes_grid:
            neighbors[2] = center_nodes_grid[right_key]

        up_key = (center_row + 2, center_col)
        if up_key in center_nodes_grid:
            neighbors[3] = center_nodes_grid[up_key]

        down_key = (center_row - 2, center_col)
        if down_key in center_nodes_grid:
            neighbors[4] = center_nodes_grid[down_key]

        return neighbors

    n_centers = len(center_nodes)
    neighbours_dict = {node: get_center_neighbors_from_grid(node, center_nodes_grid, nWidth)
                       for node in center_nodes}

    # -------- Compute Areas --------
    def heron_triangle_area(p1, p2, p3):
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        s = (a + b + c) / 2.0
        val = s * (s - a) * (s - b) * (s - c)
        return sqrt(val) if val > 0 else 0.0

    def compute_angle_from_center(node_idx, center_pos, structural_coords):
        node_pos = structural_coords[node_idx]
        dx = node_pos[0] - center_pos[0]
        dy = node_pos[1] - center_pos[1]
        return np.arctan2(dy, dx)

    def compute_block_area(center_idx, block_nodes, structural_coords):
        center_pos = structural_coords[center_idx]

        if len(block_nodes) == 9:
            counterclockwise_order = [6, 7, 8, 5, 2, 1, 0, 3]
            surrounding_indices = [block_nodes[i] for i in counterclockwise_order]
        else:
            surrounding_indices = [node for node in block_nodes if node != center_idx]
            surrounding_indices.sort(
                key=lambda node_idx: compute_angle_from_center(node_idx, center_pos, structural_coords)
            )

        total_area = 0.0
        n = len(surrounding_indices)
        for i in range(n):
            p1 = center_pos
            p2 = structural_coords[surrounding_indices[i]]
            p3 = structural_coords[surrounding_indices[(i + 1) % n]]
            total_area += heron_triangle_area(p1, p2, p3)

        return total_area

    areas_dict = {}
    for center in center_nodes:
        block = blocks[center]
        areas_dict[center] = compute_block_area(center, block, structural_coords)

    # -------- Compute Volumes --------
    volumes_dict = {center: areas_dict[center] * fin_depth for center in center_nodes}

    # -------- Center-to-center distances --------
    center_distances = {}
    for c in center_nodes:
        cx, cy = structural_coords[c]
        for neigh in neighbours_dict[c]:
            if neigh is None or neigh == c:
                continue
            nx, ny = structural_coords[neigh]
            d = np.linalg.norm([nx - cx, ny - cy])
            center_distances[(c, neigh)] = d

    # -------- Face lengths and areas --------
    face_lengths = {}
    face_areas = {}
    for c in center_nodes:
        block_c = blocks[c]
        for neigh in neighbours_dict[c]:
            if neigh is None or neigh == c:
                continue
            block_n = blocks[neigh]
            shared = list(set(block_c).intersection(set(block_n)))
            L = 0.0
            if len(shared) >= 2:
                max_len = 0.0
                for i in range(len(shared)):
                    for j in range(i + 1, len(shared)):
                        p1 = structural_coords[shared[i]]
                        p2 = structural_coords[shared[j]]
                        candidate = np.linalg.norm(p2 - p1)
                        if candidate > max_len:
                            max_len = candidate
                L = max_len
            face_lengths[(c, neigh)] = L
            face_areas[(c, neigh)] = L * fin_depth

    # -------- Boundary areas --------
    boundary_areas = {}
    for c in center_nodes:
        A_surf = 0.0
        for direction_idx, neigh in enumerate(neighbours_dict[c]):
            if neigh is None:
                block_c = blocks[c]
                axis = 0 if direction_idx in (1, 2) else 1
                coords_block = structural_coords[block_c][:, axis]

                if direction_idx in (1, 4):
                    edge_coord = np.min(coords_block)
                else:
                    edge_coord = np.max(coords_block)

                tol = 1e-12 + 1e-6 * abs(edge_coord)
                edge_nodes = [n for n in block_c if abs(structural_coords[n][axis] - edge_coord) <= tol]

                edge_len = 0.0
                if len(edge_nodes) >= 2:
                    for i in range(len(edge_nodes)):
                        for j in range(i + 1, len(edge_nodes)):
                            p1 = structural_coords[edge_nodes[i]]
                            p2 = structural_coords[edge_nodes[j]]
                            edge_len = max(edge_len, np.linalg.norm(p2 - p1))

                A_surf += edge_len * fin_depth
        boundary_areas[c] = A_surf

    return (structural_coords,
            center_nodes,
            neighbours_dict,
            areas_dict,
            volumes_dict,
            blocks,
            center_distances,
            face_lengths,
            face_areas,
            boundary_areas)


def compute_fin_mass(fin_thickness, fin_height, taper_ratio, fin_depth, rho=2700):
    """
    Compute mass of a single tapered fin.

    Parameters
    ----------
    fin_thickness : float
        Base thickness (m)
    fin_height : float
        Height (m)
    taper_ratio : float
        Tip/base thickness ratio
    fin_depth : float
        Depth (m)
    rho : float
        Density (kg/m³)

    Returns
    -------
    mass : float
        Fin mass (kg)
    """
    # Trapezoidal cross-section area
    # Average width = (base_width + tip_width) / 2
    avg_width = fin_thickness * (1 + taper_ratio) / 2
    cross_section_area = avg_width * fin_height
    volume = cross_section_area * fin_depth
    mass = rho * volume
    return mass


if __name__ == "__main__":
    # Test with different geometries
    print("Testing parametric fin geometry...")

    # Rectangular fin
    print("\n1. Rectangular fin (taper_ratio=1.0):")
    geom1 = build_parametric_fin_grid(fin_thickness=0.002, fin_height=0.025, taper_ratio=1.0)
    mass1 = compute_fin_mass(0.002, 0.025, 1.0, 0.050)
    print(f"   Mass: {mass1*1000:.2f} g")

    # Tapered fin
    print("\n2. Tapered fin (taper_ratio=0.5):")
    geom2 = build_parametric_fin_grid(fin_thickness=0.002, fin_height=0.025, taper_ratio=0.5)
    mass2 = compute_fin_mass(0.002, 0.025, 0.5, 0.050)
    print(f"   Mass: {mass2*1000:.2f} g")

    print(f"\n   Mass reduction: {(1 - mass2/mass1)*100:.1f}%")
