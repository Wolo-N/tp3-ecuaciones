# -*- coding: utf-8 -*-
"""
2D Finite Volume Method Grid Generator

Based on Session21 slide 9 exercise:
Build the grid matrices using Q9 shape functions for coordinate transformation

**Grid Components**:
- Nodes: Physical coordinates (x, y) of grid points
- Neighbors: Connectivity information (N, S, E, W, NE, NW, SE, SW)
- Areas: Interface areas between control volumes
- Volumes: Control volume sizes for each node
"""

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt

# ---------------- Shape functions (Q9) ----------------
def q9_shape_functions(xi, eta):
    """
    Q9 (9-node quadrilateral) shape functions
    Node ordering: 1-4 corners, 5-8 edges, 9 center
    """
    # 1D quadratic Lagrange polynomials
    L1 = 0.5 * xi * (xi - 1.0)
    L2 = 1.0 - xi**2
    L3 = 0.5 * xi * (xi + 1.0)

    M1 = 0.5 * eta * (eta - 1.0)
    M2 = 1.0 - eta**2
    M3 = 0.5 * eta * (eta + 1.0)

    # Tensor-product shape functions (Q9)
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

# --------------- Interpolation -------------------
def q9_interpolate_points(ctrl_pts, natural_coords):
    """
    Interpolate physical coordinates using Q9 shape functions

    Parameters
    ----------
    ctrl_pts : (9,2) array
        Control points in physical space [x, y]
    natural_coords : (m,2) array
        Natural coordinates (xi, eta) in [-1,1]^2

    Returns
    -------
    physical_coords : (m,2) array
        Interpolated physical coordinates
    """
    ctrl_pts = np.asarray(ctrl_pts, dtype=float).reshape(9, 2)
    natural_coords = np.asarray(natural_coords, dtype=float).reshape(-1, 2)

    # Build matrix of shape functions for all query points
    Nmat = np.vstack([q9_shape_functions(xi, eta) for xi, eta in natural_coords])

    # Interpolate all points at once
    physical_coords = Nmat @ ctrl_pts
    return physical_coords

# -------- Grid Generation Functions --------
def build_2d_grid(ctrl_pts, nHeight, nWidth, include_centers=True):
    """
    Build 2D structured grid with FVM matrices following Session 19-21 material

    Creates a grid with:
    - Regular nodes (border/corner nodes of control volumes)
    - Center nodes (at the center of each control volume/block)
    - Blocks (quadrilateral control volumes)

    Parameters
    ----------
    ctrl_pts : (9,2) array
        Control points defining the domain geometry
    nHeight : int
        Number of blocks in eta direction
    nWidth : int
        Number of blocks in xi direction
    include_centers : bool
        If True, includes center nodes for each block (default: True)

    Returns
    -------
    regular_nodes : (nRegular, 2) array
        Physical coordinates of regular (border) nodes
    center_nodes : (nCenters, 2) array
        Physical coordinates of center nodes (if include_centers=True)
    blocks : (nBlocks, 4) array
        Block connectivity: indices of corner nodes [SW, SE, NE, NW]
    neighbours : (nBlocks, 9) array
        Block neighbor indices [W, E, S, N, SW, SE, NW, NE, P]
        None for blocks outside domain
    areas : (nBlocks, 4) array
        Interface areas [A_w, A_e, A_s, A_n]
    volumes : (nBlocks,) array
        Control volume for each block
    """
    # Natural coordinates (uniform spacing)
    xi_coords = np.linspace(-1, 1, nWidth)
    eta_coords = np.linspace(-1, 1, nHeight)

    # Spacing in natural coordinates
    dxi = xi_coords[1] - xi_coords[0] if nWidth > 1 else 0
    deta = eta_coords[1] - eta_coords[0] if nHeight > 1 else 0

    # Create meshgrid
    Xi, Eta = np.meshgrid(xi_coords, eta_coords, indexing='ij')
    natural_coords = np.column_stack((Xi.ravel(), Eta.ravel()))

    # Map to physical coordinates
    nodes = q9_interpolate_points(ctrl_pts, natural_coords)

    nTotal = nWidth * nHeight

    # Initialize arrays
    neighbours = np.full((nTotal, 9), None, dtype=object)
    areas = np.zeros((nTotal, 4))  # [A_w, A_e, A_s, A_n]
    volumes = np.zeros(nTotal)

    # Helper function to get node index
    def idx(i, j):
        """Convert (i,j) to linear index"""
        if 0 <= i < nWidth and 0 <= j < nHeight:
            return i * nHeight + j
        return None

    # Build connectivity and compute geometric quantities
    for i in range(nWidth):
        for j in range(nHeight):
            n = idx(i, j)

            # Neighbor indices: [W, E, S, N, SW, SE, NW, NE, P]
            neighbours[n, 0] = idx(i-1, j)      # W
            neighbours[n, 1] = idx(i+1, j)      # E
            neighbours[n, 2] = idx(i, j-1)      # S
            neighbours[n, 3] = idx(i, j+1)      # N
            neighbours[n, 4] = idx(i-1, j-1)    # SW
            neighbours[n, 5] = idx(i+1, j-1)    # SE
            neighbours[n, 6] = idx(i-1, j+1)    # NW
            neighbours[n, 7] = idx(i+1, j+1)    # NE
            neighbours[n, 8] = n                # P (self)

            # Compute interface areas and volume
            # Areas are computed at cell faces
            x_p, y_p = nodes[n]

            # West face area
            if neighbours[n, 0] is not None:
                x_w, y_w = nodes[neighbours[n, 0]]
                # Distance between nodes
                dx_w = abs(x_p - x_w)
                # For 2D planar: area = length * unit_depth (assume depth = 1)
                # Use average dy spacing
                dy = deta * 0.5 * (abs(y_p - nodes[idx(i, max(0, j-1))][1] if idx(i, max(0, j-1)) else y_p) +
                                    abs(nodes[idx(i, min(nHeight-1, j+1))][1] - y_p if idx(i, min(nHeight-1, j+1)) else y_p))
                areas[n, 0] = dy if dy > 0 else deta

            # East face area
            if neighbours[n, 1] is not None:
                x_e, y_e = nodes[neighbours[n, 1]]
                dx_e = abs(x_e - x_p)
                dy = deta * 0.5 * (abs(y_p - nodes[idx(i, max(0, j-1))][1] if idx(i, max(0, j-1)) else y_p) +
                                    abs(nodes[idx(i, min(nHeight-1, j+1))][1] - y_p if idx(i, min(nHeight-1, j+1)) else y_p))
                areas[n, 1] = dy if dy > 0 else deta

            # South face area
            if neighbours[n, 2] is not None:
                x_s, y_s = nodes[neighbours[n, 2]]
                dy_s = abs(y_p - y_s)
                dx = dxi * 0.5 * (abs(x_p - nodes[idx(max(0, i-1), j)][0] if idx(max(0, i-1), j) else x_p) +
                                   abs(nodes[idx(min(nWidth-1, i+1), j)][0] - x_p if idx(min(nWidth-1, i+1), j) else x_p))
                areas[n, 2] = dx if dx > 0 else dxi

            # North face area
            if neighbours[n, 3] is not None:
                x_n, y_n = nodes[neighbours[n, 3]]
                dy_n = abs(y_n - y_p)
                dx = dxi * 0.5 * (abs(x_p - nodes[idx(max(0, i-1), j)][0] if idx(max(0, i-1), j) else x_p) +
                                   abs(nodes[idx(min(nWidth-1, i+1), j)][0] - x_p if idx(min(nWidth-1, i+1), j) else x_p))
                areas[n, 3] = dx if dx > 0 else dxi

            # Control volume (area in 2D)
            # Approximate as rectangle using average spacing
            dx_vol = dxi
            dy_vol = deta

            # Better approximation using neighbors
            if neighbours[n, 0] is not None and neighbours[n, 1] is not None:
                dx_vol = 0.5 * (abs(nodes[neighbours[n, 1]][0] - x_p) + abs(x_p - nodes[neighbours[n, 0]][0]))
            elif neighbours[n, 1] is not None:
                dx_vol = abs(nodes[neighbours[n, 1]][0] - x_p)
            elif neighbours[n, 0] is not None:
                dx_vol = abs(x_p - nodes[neighbours[n, 0]][0])

            if neighbours[n, 2] is not None and neighbours[n, 3] is not None:
                dy_vol = 0.5 * (abs(nodes[neighbours[n, 3]][1] - y_p) + abs(y_p - nodes[neighbours[n, 2]][1]))
            elif neighbours[n, 3] is not None:
                dy_vol = abs(nodes[neighbours[n, 3]][1] - y_p)
            elif neighbours[n, 2] is not None:
                dy_vol = abs(y_p - nodes[neighbours[n, 2]][1])

            volumes[n] = dx_vol * dy_vol

    return nodes, neighbours, areas, volumes, xi_coords, eta_coords

# -------- Visualization Functions --------
def plot_grid(nodes, neighbours, nWidth, nHeight, show_indices=True):
    """
    Plot the 2D structured grid

    Parameters
    ----------
    nodes : (nTotal, 2) array
        Node coordinates
    neighbours : (nTotal, 9) array
        Neighbor connectivity
    nWidth : int
        Number of nodes in xi direction
    nHeight : int
        Number of nodes in eta direction
    show_indices : bool
        Whether to show node indices
    """
    plt.figure(figsize=(10, 8))

    # Plot grid lines
    for i in range(nWidth):
        for j in range(nHeight):
            n = i * nHeight + j
            x_p, y_p = nodes[n]

            # Draw lines to E and N neighbors
            if neighbours[n, 1] is not None:  # E
                x_e, y_e = nodes[neighbours[n, 1]]
                plt.plot([x_p, x_e], [y_p, y_e], 'b-', linewidth=0.5)

            if neighbours[n, 3] is not None:  # N
                x_n, y_n = nodes[neighbours[n, 3]]
                plt.plot([x_p, x_n], [y_p, y_n], 'b-', linewidth=0.5)

    # Plot nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], s=30, c='red', marker='o', zorder=5)

    # Show node indices
    if show_indices:
        for i, (x, y) in enumerate(nodes):
            plt.text(x + 0.02, y + 0.02, str(i), fontsize=7, color='darkred')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Structured Grid for FVM')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

def print_grid_info(nodes, neighbours, areas, volumes):
    """Print summary information about the grid"""
    print("="*60)
    print("2D FVM GRID SUMMARY")
    print("="*60)
    print(f"Total nodes: {len(nodes)}")
    print(f"Grid dimensions: {nodes.shape}")
    print(f"\nNode coordinates (first 5):")
    print(nodes[:5])
    print(f"\nNeighbours matrix shape: {neighbours.shape}")
    print(f"  Columns: [W, E, S, N, SW, SE, NW, NE, P]")
    print(f"\nFirst node neighbors (node 0):")
    print(f"  W={neighbours[0,0]}, E={neighbours[0,1]}, S={neighbours[0,2]}, N={neighbours[0,3]}")
    print(f"  SW={neighbours[0,4]}, SE={neighbours[0,5]}, NW={neighbours[0,6]}, NE={neighbours[0,7]}, P={neighbours[0,8]}")
    print(f"\nAreas matrix shape: {areas.shape}")
    print(f"  Columns: [A_w, A_e, A_s, A_n]")
    print(f"  First node areas: {areas[0]}")
    print(f"\nVolumes array shape: {volumes.shape}")
    print(f"  First 5 volumes: {volumes[:5]}")
    print(f"  Total volume: {volumes.sum():.6f}")
    print("="*60)

# ============ MAIN EXAMPLE ============
if __name__ == "__main__":
    print("2D FVM Grid Generator - Session 21, Slide 9 Exercise\n")

    # Grid resolution
    nHeight = 11
    nWidth = 11

    print(f"Creating {nWidth}x{nHeight} structured grid...\n")

    # Control points for Q9 mapping (example: unit square)
    # For a simple rectangular domain
    ctrl = np.array([
        [-1.0, -1.0],    # N1  (-1,-1)
        [ 1.0, -1.0],    # N2  ( 1,-1)
        [ 1.0,  1.0],    # N3  ( 1, 1)
        [-1.0,  1.0],    # N4  (-1, 1)
        [ 0.0, -1.0],    # N5  ( 0,-1)
        [ 1.0,  0.0],    # N6  ( 1, 0)
        [ 0.0,  1.0],    # N7  ( 0, 1)
        [-1.0,  0.0],    # N8  (-1, 0)
        [ 0.0,  0.0]     # N9  ( 0, 0)
    ])

    # Build grid matrices
    nodes, neighbours, areas, volumes, xi_coords, eta_coords = build_2d_grid(ctrl, nHeight, nWidth)

    # Print information
    print_grid_info(nodes, neighbours, areas, volumes)

    # Visualize
    plot_grid(nodes, neighbours, nWidth, nHeight, show_indices=True)
    plt.savefig('2d_fvm_grid.png', dpi=150, bbox_inches='tight')
    print("\nGrid visualization saved as '2d_fvm_grid.png'")
    plt.show()

    # Example: Access specific node information
    print("\nExample - Node 60 (center node):")
    n = 60
    print(f"  Coordinates: ({nodes[n,0]:.3f}, {nodes[n,1]:.3f})")
    print(f"  Neighbors: W={neighbours[n,0]}, E={neighbours[n,1]}, S={neighbours[n,2]}, N={neighbours[n,3]}")
    print(f"  Areas: A_w={areas[n,0]:.4f}, A_e={areas[n,1]:.4f}, A_s={areas[n,2]:.4f}, A_n={areas[n,3]:.4f}")
    print(f"  Volume: {volumes[n]:.4f}")
