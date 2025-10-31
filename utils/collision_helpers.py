import numpy as np 
import torch

def compute_corners(positions, headings, lengths, widths):
    """ Compute the four corners of bounding boxes in batches for multiple agents and timesteps."""
    # Half dimensions
    dx = lengths / 2  # [A, T]
    dy = widths / 2  # [A, T]

    # Define corner offsets relative to center position
    corner_offsets = np.array([
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1]
    ])  # [4, 2]

    # Scale offsets by dx and dy
    corner_offsets = corner_offsets[None, None, :, :] * np.stack([dx, dy], axis=-1)[:, :, None, :]  # [A, T, 4, 2]

    # Rotation matrices from headings
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    rotation_matrix = np.stack([
        np.stack([cos_h, -sin_h], axis=-1),
        np.stack([sin_h, cos_h], axis=-1)
    ], axis=-2)  # [A, T, 2, 2]

    # Rotate corners and translate to positions
    rotated_corners = np.einsum('atij,atfj->atfi', rotation_matrix, corner_offsets)  # [A, T, 4, 2]  # [A, T, 4, 2]
    translated_rotated_corners = rotated_corners + positions[:, :, None, :]  # [A, T, 4, 2]

    return translated_rotated_corners


def get_axes(vertices):
    """ Computes the normalized normal (perpendicular) axes of the edges of a polygon."""
    n = vertices.shape[0]
    indices = np.arange(n)
    next_indices = (indices + 1) % n
    edges = vertices[next_indices] - vertices[indices]
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    lengths = np.hypot(normals[:, 0], normals[:, 1])
    # Avoid division by zero for degenerate edges
    lengths[lengths == 0] = 1
    axes = normals / lengths[:, np.newaxis]
    return axes


def is_colliding(poly1, poly2):
    """ Determines if two convex quadrilaterals are colliding using the Separating Axis Theorem."""
    # Collect axes from both polygons
    axes1 = get_axes(poly1)  # Shape: (n1, 2)
    axes2 = get_axes(poly2)  # Shape: (n2, 2)
    axes = np.vstack((axes1, axes2))  # Combined axes, shape: (n1 + n2, 2)

    # Project both polygons onto all axes
    projections1 = np.dot(poly1, axes.T)  # Shape: (4, n_axes)
    projections2 = np.dot(poly2, axes.T)  # Shape: (4, n_axes)

    # Find the min and max projections for each polygon on each axis
    min1 = projections1.min(axis=0)
    max1 = projections1.max(axis=0)
    min2 = projections2.min(axis=0)
    max2 = projections2.max(axis=0)

    # Check for any separating axis
    separated = (max1 < min2) | (max2 < min1)
    if np.any(separated):
        # Separating axis found; polygons are not colliding
        return False
    else:
        # No separating axis found; polygons are colliding
        return True
    

def batched_collision_checker(ego_state, agent_states):
    """ Perform batched collision checking for ego and agent states."""
    A, T = agent_states.shape[0], agent_states.shape[1]
    collision = np.zeros((A, T), dtype=int)
    
    # Split states into components
    ego_pos = ego_state[:, :, :2]  # [1, 90, 2]
    ego_heading = ego_state[:, :, 2]  # [1, 90]
    ego_length = ego_state[:, :, 3]  # [1, 90]
    ego_width = ego_state[:, :, 4]  # [1, 90]

    agent_pos = agent_states[:, :, :2]  # [A, 90, 2]
    agent_heading = agent_states[:, :, 2]  # [A, 90]
    agent_length = agent_states[:, :, 3]  # [A, 90]
    agent_width = agent_states[:, :, 4]  # [A, 90]

    # Compute corners of bounding boxes
    agent_corners_all = compute_corners(agent_pos, agent_heading, agent_length, agent_width)  # [A, 90, 4, 2]
    ego_corners_all = compute_corners(ego_pos, ego_heading, ego_length, ego_width)  # [1, 90, 4, 2]

    # Iterate over each timestep
    for t in range(T):
        # Ego bounding box for the current timestep
        ego_corners = ego_corners_all[0, t]
        
        # Agent bounding boxes for the current timestep
        for a in range(A):
            agent_corners = agent_corners_all[a, t]
            # Check for overlap
            if is_colliding(ego_corners, agent_corners):
                collision[a, t] = 1  # Mark as collision

    return collision.astype(int)