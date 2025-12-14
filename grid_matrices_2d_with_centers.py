import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
from parameters import N_HEIGHT, N_WIDTH


# ---------------- Shape functions ----------------
def q9_shape_functions(xi, eta):
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
    Interpolate many physical coordinates (X,Y) using Q9 shape functions.

    Parameters
    ----------
    ctrl_pts : (9,2) array_like
        Control-point coordinates ordered as:
        [(-1,-1), ( 1,-1), ( 1, 1), (-1, 1),
         ( 0,-1), ( 1, 0), ( 0, 1), (-1, 0), ( 0, 0)]
    natural_coords : (m,2) array_like
        Each row is (xi, eta) in [-1,1]^2.

    Returns
    -------
    structural_coords : (m,2) ndarray
        Interpolated points.
    """
    ctrl_pts = np.asarray(ctrl_pts, dtype=float).reshape(9, 2)
    natural_coords = np.asarray(natural_coords, dtype=float).reshape(-1, 2)

    # Build matrix of shape functions for all query points
    Nmat = np.vstack([q9_shape_functions(xi, eta) for xi, eta in natural_coords])  # (m,9)

    # Interpolate all points at once
    structural_coords = Nmat @ ctrl_pts  # (m,2)
    return structural_coords

# -------- Select center nodes (every other node in both x and y) --------
def get_center_nodes(nHeight, nWidth):
    """
    Select center nodes from the grid by taking every other node in both directions.

    For an 11x11 grid, this selects nodes at rows 0, 2, 4, 6, 8, 10 and
    columns 0, 2, 4, 6, 8, 10, creating a 6x6 grid of center nodes.

    Parameters
    ----------
    nHeight : int
        Number of nodes in eta direction (rows)
    nWidth : int
        Number of nodes in xi direction (columns)

    Returns
    -------
    center_nodes : list
        List of global node indices for center nodes
    center_nodes_grid : dict
        Dictionary mapping (row, col) -> global index for center nodes
    """
    center_nodes = []
    center_nodes_grid = {}  # Map (row, col) -> global index for center nodes

    for row in range(0, nHeight, 2):  # rows 0, 2, 4, 6, 8, 10
        for col in range(0, nWidth, 2):  # cols 0, 2, 4, 6, 8, 10
            node_index = row * nWidth + col
            center_nodes.append(node_index)
            center_nodes_grid[(row, col)] = node_index

    return center_nodes, center_nodes_grid


# -------- Define blocks (3x3 neighborhoods) around each center node --------
def get_block_nodes(center_idx, nWidth, nHeight):
    """
    Get the 9 nodes (3x3 block) around a center node.
    Returns list of node indices in the block.
    """
    # Convert center index to row, col
    center_row = center_idx // nWidth
    center_col = center_idx % nWidth

    block = []
    # Get 3x3 neighborhood (including the center)
    for dr in [-1, 0, 1]:  # row offset
        for dc in [-1, 0, 1]:  # column offset
            r = center_row + dr
            c = center_col + dc
            # Check if within grid bounds
            if 0 <= r < nHeight and 0 <= c < nWidth:
                block.append(r * nWidth + c)

    return block


# -------- Build neighbor connectivity for center nodes --------
def get_center_neighbors_from_grid(center_idx, center_nodes_grid, nWidth):
    """
    Get the neighbors of a center node using the grid structure in the format:
    [Center, Left, Right, Up, Down]

    NOTE: Due to how meshgrid is set up:
        - row index maps to X coordinate (fin thickness direction)
        - col index maps to Y coordinate (fin height direction)

    Parameters
    ----------
    center_idx : int
        Global grid index of the center node
    center_nodes_grid : dict
        Dictionary mapping (row, col) -> global index for center nodes
    nWidth : int
        Width of the full grid

    Returns
    -------
    neighbors : list
        [Center, Left, Right, Up, Down] where None indicates no neighbor
    """
    # Convert center index to row, col in the global grid
    center_row = center_idx // nWidth
    center_col = center_idx % nWidth

    # Initialize neighbors: [Center, Left, Right, Up, Down]
    neighbors = [center_idx, None, None, None, None]

    # Left neighbor: row - 2 (lower X), same col
    left_key = (center_row - 2, center_col)
    if left_key in center_nodes_grid:
        neighbors[1] = center_nodes_grid[left_key]

    # Right neighbor: row + 2 (higher X), same col
    right_key = (center_row + 2, center_col)
    if right_key in center_nodes_grid:
        neighbors[2] = center_nodes_grid[right_key]

    # Up neighbor: same row, col + 2 (higher Y)
    up_key = (center_row, center_col + 2)
    if up_key in center_nodes_grid:
        neighbors[3] = center_nodes_grid[up_key]

    # Down neighbor: same row, col - 2 (lower Y)
    down_key = (center_row, center_col - 2)
    if down_key in center_nodes_grid:
        neighbors[4] = center_nodes_grid[down_key]

    return neighbors


# -------- Compute Areas using Heron's Formula --------
def heron_triangle_area(p1, p2, p3):
    """
    Calculate the area of a triangle using Heron's formula.

    Parameters
    ----------
    p1, p2, p3 : array_like
        Coordinates of the three vertices of the triangle

    Returns
    -------
    area : float
        Area of the triangle
    """
    # Calculate side lengths
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)

    # Semi-perimeter
    s = (a + b + c) / 2.0

    # Heron's formula
    val = s * (s - a) * (s - b) * (s - c)
    if val <= 0:
        return 0.0  # Degenerate or collinear triangle; no area
    area = sqrt(val)

    return area

def _compute_angle_from_center(node_idx, center_pos, structural_coords):
    """
    Helper function to compute the angle from center to a node.

    Parameters
    ----------
    node_idx : int
        Global index of the node
    center_pos : ndarray
        Position of the center node
    structural_coords : ndarray
        Coordinates of all nodes

    Returns
    -------
    angle : float
        Angle in radians from center to node
    """
    node_pos = structural_coords[node_idx]
    dx = node_pos[0] - center_pos[0]
    dy = node_pos[1] - center_pos[1]
    return np.arctan2(dy, dx)

def compute_block_area(center_idx, block_nodes, structural_coords):
    """
    Compute the area of a block by splitting it into triangles.
    Each triangle has the center node as one vertex.

    The block is split by connecting the center node to all surrounding nodes,
    forming triangles with consecutive pairs of surrounding nodes.

    Parameters
    ----------
    center_idx : int
        Global index of the center node
    block_nodes : list
        List of node indices in the 3x3 block
    structural_coords : ndarray
        Coordinates of all nodes

    Returns
    -------
    total_area : float
        Total area of the block
    """
    # Get center node position
    center_pos = structural_coords[center_idx]

    # Get surrounding nodes in counterclockwise order from block_nodes
    # Block nodes are ordered row-by-row, we need to reorder them counterclockwise
    # Standard 3x3 block layout (row-by-row):
    # [0, 1, 2]   --> corresponds to: [TL, T, TR]
    # [3, 4, 5]   --> corresponds to: [L,  C, R ]
    # [6, 7, 8]   --> corresponds to: [BL, B, BR]

    # Create mapping from row-by-row to counterclockwise order around center
    # Counterclockwise starting from bottom-left: BL, B, BR, R, TR, T, TL, L
    if len(block_nodes) == 9:
        # Full block with all 9 nodes
        counterclockwise_order = [6, 7, 8, 5, 2, 1, 0, 3]  # Indices in block_nodes list
        surrounding_indices = [block_nodes[i] for i in counterclockwise_order]
    else:
        # Boundary block - extract surrounding nodes (exclude center)
        surrounding_indices = [node for node in block_nodes if node != center_idx]

        # Sort surrounding nodes counterclockwise based on their position
        surrounding_indices.sort(
            key=lambda node_idx: _compute_angle_from_center(node_idx, center_pos, structural_coords)
        )

    # Calculate total area by summing triangles
    total_area = 0.0
    n = len(surrounding_indices)

    for i in range(n):
        p1 = center_pos
        p2 = structural_coords[surrounding_indices[i]]
        p3 = structural_coords[surrounding_indices[(i + 1) % n]] #para conectarlo con el indice 0 y de toda la vuelta

        triangle_area = heron_triangle_area(p1, p2, p3)
        total_area += triangle_area

    return total_area


# ---------------- Plot ----------------
def plot_grid_with_blocks(structural_coords, center_nodes, nHeight, nWidth, x_exaggeration=5.0):
    """
    Plot the grid showing all nodes, center nodes, and blocks.

    Parameters
    ----------
    structural_coords : ndarray
        Coordinates of all nodes in the grid
    center_nodes : list
        List of global node indices for center nodes
    nHeight : int
        Number of nodes in eta direction (rows)
    nWidth : int
        Number of nodes in xi direction (columns)
    x_exaggeration : float, optional
        Factor to stretch the x-axis for visualization so the thin fin is visible.
    """

    # Exaggerate x for plotting only to make the fin thickness visible
    scaled_coords = structural_coords.copy()
    scaled_coords[:, 0] *= x_exaggeration

    plt.figure(figsize=(10, 8))

    # Plot all nodes first (in blue)
    plt.scatter(scaled_coords[:, 0], scaled_coords[:, 1], s=20, marker='o',
                color='blue', label='Regular nodes', zorder=3)

    # Plot center nodes in a different color (red)
    center_coords = scaled_coords[center_nodes]
    plt.scatter(center_coords[:, 0], center_coords[:, 1], s=50, marker='o',
                color='red', label='Center nodes', zorder=5)

    # Draw rectangles around each block
    for center_idx in center_nodes:
        # For a 3x3 block, get the min/max row and column of the block
        center_row = center_idx // nWidth
        center_col = center_idx % nWidth

        # Determine actual bounds of the block
        min_row = max(0, center_row - 1)
        max_row = min(nHeight - 1, center_row + 1)
        min_col = max(0, center_col - 1)
        max_col = min(nWidth - 1, center_col + 1)

        # Get the four corners of the bounding box in counterclockwise order
        corners_idx = [
            min_row * nWidth + min_col,  # bottom-left
            min_row * nWidth + max_col,  # bottom-right
            max_row * nWidth + max_col,  # top-right
            max_row * nWidth + min_col   # top-left
        ]

        # Get coordinates of corners
        corners = [scaled_coords[idx] for idx in corners_idx]

        # Draw polygon
        polygon = Polygon(corners, fill=False, edgecolor='green',
                        linewidth=1.5, linestyle='-', alpha=0.6)
        plt.gca().add_patch(polygon)

    # Add labels for all nodes
    for i, (x, y) in enumerate(scaled_coords):
        # Use different color for center nodes labels
        if i in center_nodes:
            plt.text(x + 1e-3, y + 1e-3, str(i), fontsize=7, color='red', fontweight='bold', zorder=6)
        else:
            plt.text(x + 1e-3, y + 1e-3, str(i), fontsize=7, color='blue', zorder=6)

    # Add custom legend entry for blocks
    block_patch = mpatches.Patch(facecolor='none', edgecolor='green', linewidth=1.5, label='Blocks')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(block_patch)
    labels.append('Blocks')


    plt.title('Structural Grid Points with Blocks')
    plt.xlabel(f'Width (X) [exaggerated x{ x_exaggeration }]')
    plt.ylabel('Height (Y)')
    plt.legend(handles=handles, labels=labels)
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def build_fin_grid_2d(fin_thickness=0.002,
                      fin_height=0.025,
                      fin_length=0.050,
                      tip_ratio=1.0,
                      round_factor=0.5):
    """
    Builds the 2D fin grid and all FVM geometric data
    for the given fin_thickness, fin_height and fin_length.
    """


    # -------- Discretization in natural coordinates --------
    nHeight = N_HEIGHT  # Number of nodes in eta direction (should be odd for proper center nodes)
    nWidth  = N_WIDTH   # Number of nodes in xi direction (should be odd for proper center nodes)


    h = np.linspace(-1, 1, nHeight)  # eta
    w = np.linspace(-1, 1, nWidth)   # xi
    H, W = np.meshgrid(h, w)          # default 'xy': W->xi, H->eta

    # (xi, eta) rows
    natural_coords = np.column_stack((W.ravel(), H.ravel()))

    # -------- Control points for symmetric tapered fin --------
    # Fin shape: flat base at bottom (Y=-25mm), tapered tip at top (Y=+25mm)
    # Symmetric about X=0
    # Base width: 50mm, Tip width: 30mm, Height: 50mm
    # ---- Fin geometry (rectangular) ----

    half_t_base = fin_thickness / 2.0
    half_t_tip  = tip_ratio * half_t_base

    # Punta redondeada: las esquinas superiores bajan respecto al centro
    # round_factor = 0 → punta plana/aguda (todas las esquinas a la misma altura)
    # round_factor = 1 → punta redondeada (esquinas bajan 10% de la altura)
    tip_corner_drop = round_factor * fin_height * 0.1
    tip_corner_height = fin_height - tip_corner_drop

    # Ancho a mitad de altura (promedio lineal entre base y punta)
    half_t_mid = 0.5 * (half_t_base + half_t_tip)


    ctrl = np.array([
        # Base (recta)
        [-half_t_base, 0.0],                # N1: base izquierda
        [ half_t_base, 0.0],                # N2: base derecha

        # Esquinas de la punta (bajan según round_factor para redondear)
        [ half_t_tip,  tip_corner_height],  # N3: esquina superior derecha
        [-half_t_tip,  tip_corner_height],  # N4: esquina superior izquierda

        # Puntos intermedios
        [ 0.0,         0.0],                # N5: centro base

        # Mitad de altura
        [ half_t_mid,  fin_height/2],       # N6: mitad derecha
        [ 0.0,         fin_height],         # N7: centro punta (punto más alto)
        [-half_t_mid,  fin_height/2],       # N8: mitad izquierda
        [ 0.0,         fin_height/2],       # N9: centro medio
    ])




    # -------- Coordinate transformation --------
    structural_coords = q9_interpolate_points(ctrl, natural_coords)

    center_nodes, center_nodes_grid = get_center_nodes(nHeight, nWidth)

    # Create dictionary mapping center nodes to their blocks
    blocks = {}
    for center in center_nodes:
        blocks[center] = get_block_nodes(center, nWidth, nHeight)

    # Build neighbor array for all center nodes
    n_centers = len(center_nodes)
    neighbours = np.empty((n_centers, 5), dtype=object)

    for i, center in enumerate(center_nodes):
        neighbours[i, :] = get_center_neighbors_from_grid(center, center_nodes_grid, nWidth)

    # Create a dictionary for easy lookup: global_node_index -> neighbor list
    # This allows you to quickly find neighbors by the global node index
    center_node_to_idx = {node: i for i, node in enumerate(center_nodes)}
    neighbours_dict = {node: neighbours[i] for i, node in enumerate(center_nodes)}

    # Compute areas for all center nodes
    areas = np.zeros(n_centers, dtype=float)
    areas_dict = {}

    for i, center in enumerate(center_nodes):
        block = blocks[center]
        areas[i] = compute_block_area(center, block, structural_coords)
        areas_dict[center] = areas[i]

     #-------- Compute Volumes --------
    # For 2D grid representing 3D object: Volume = Area × Thickness
    # Heat sink base thickness: 1mm (0.001 m)
    # ---- Compute the 3D volumes ----
    # 2D grid = (thickness × height); extrude into page by fin_length

    volumes = areas * fin_length

    volumes_dict = {center_nodes[i]: volumes[i] for i in range(len(center_nodes))}


    # -----------------------------------------------------------
    # 1) CENTER-TO-CENTER DISTANCES d_ij
    # -----------------------------------------------------------
    center_distances = {}   # key: (i, j) using GLOBAL center node ids

    for c_idx, c in enumerate(center_nodes):
        cx, cy = structural_coords[c]
        for neigh in neighbours_dict[c]:
            if neigh is None or neigh == c:
                continue
            nx, ny = structural_coords[neigh]
            d = np.linalg.norm([nx - cx, ny - cy])
            center_distances[(c, neigh)] = d


    # -----------------------------------------------------------
    # 2) FACE LENGTHS AND FACE AREAS
    # -----------------------------------------------------------
    # Each control volume has a block (list of 8 surrounding node ids)
    # We find shared edges between blocks.

    face_lengths = {}   # 2D length of face between CVs (meters)
    face_areas   = {}   # 3D face area = face_length * fin_length

    for c_idx, c in enumerate(center_nodes):
        block_c = blocks[c]  # nodes surrounding CV c
    
        for neigh in neighbours_dict[c]:
            if neigh is None or neigh == c:
                continue
        
            block_n = blocks[neigh]
        
            # Find shared nodes between block polygons
            shared = list(set(block_c).intersection(set(block_n)))
            L = 0.0

            if len(shared) >= 2:
                # Choose the longest edge among the shared nodes (handles 3-node overlaps cleanly)
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
            face_areas[(c, neigh)]   = L * fin_length   # <- 3D face area

    # -----------------------------------------------------------
    # 3) BOUNDARY SURFACE AREAS (FOR CONVECTION)
    # -----------------------------------------------------------
    boundary_areas = {}  # exposed area for each boundary CV

    for c_idx, c in enumerate(center_nodes):
        A_surf = 0.0   # accumulate surface area

        # Add FRONT and BACK face areas (in Z direction)
        # These faces are created when extruding the 2D geometry by fin_length
        # Both front (Z=0) and back (Z=fin_length) are exposed to air
        A_surf += 2 * areas_dict[c]  # front + back areas

        for direction_idx, neigh in enumerate(neighbours_dict[c]):
            if neigh is None:
                # boundary face: find the corresponding face length
                block_c = blocks[c]

                # Determine boundary edge by selecting nodes on the outermost coordinate
                # direction_idx: 0=Center, 1=Left, 2=Right, 3=Up, 4=Down
                # Left/Right (1,2): boundaries in X direction (axis 0)
                # Up/Down (3,4): boundaries in Y direction (axis 1)
                axis = 0 if direction_idx in (1, 2) else 1
                coords_block = structural_coords[block_c][:, axis]

                # Left (1) or Down (4) → minimum edge
                # Right (2) or Up (3) → maximum edge
                if direction_idx in (1, 4):
                    edge_coord = np.min(coords_block)
                else:
                    edge_coord = np.max(coords_block)

                tol = 1e-12 + 1e-6 * abs(edge_coord)
                edge_nodes = [n for n in block_c if abs(structural_coords[n][axis] - edge_coord) <= tol]

                # Pick the longest segment among edge nodes (covers >2 collinear nodes)
                edge_len = 0.0
                if len(edge_nodes) >= 2:
                    for i in range(len(edge_nodes)):
                        for j in range(i + 1, len(edge_nodes)):
                            p1 = structural_coords[edge_nodes[i]]
                            p2 = structural_coords[edge_nodes[j]]
                            edge_len = max(edge_len, np.linalg.norm(p2 - p1))

                # Convert to 3D area: edge x fin_length
                A_surf += edge_len * fin_length
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
