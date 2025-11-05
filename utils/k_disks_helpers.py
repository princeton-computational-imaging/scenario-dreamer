import numpy as np
from utils.geometry import normalize_angle
from tqdm import tqdm

def compute_k_disks(state_transitions, vocab_size, l, w, eps):
    box_coords = np.array([
        [-l/2, -w/2],
        [-l/2, w/2],
        [l/2, w/2],
        [l/2, -w/2]
    ])

    print("Number of state transitions: ", len(state_transitions))
    V = []
    for i in tqdm(range(vocab_size)):
        print("Number of state transitions: ", len(state_transitions))
        indices = np.arange(len(state_transitions))
        rand_idx = np.random.choice(indices, 1)[0]
        # ensure sampled transition is within reasonable bounds
        while not(state_transitions[rand_idx, 0] > -0.2 and 
                  state_transitions[rand_idx, 0] < 3.5 and 
                  np.abs(state_transitions[rand_idx, 1]) < 0.25):
            rand_idx = np.random.choice(indices, 1)[0]
        
        V.append(state_transitions[rand_idx])

        # first apply rotation
        box_coords_duplicated = np.tile(box_coords[None, :, :], (len(state_transitions), 1, 1))
        box_coords_duplicated = box_coords_duplicated.reshape(-1, 2)

        rotations = state_transitions[:, 2].repeat(4) # 4 for the four corners of each box
        cos_theta = np.cos(rotations)
        sin_theta = np.sin(rotations)

        # Create rotation matrices
        rotation_matrices = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])  # Shape [2, 2, N]
        rotation_matrices = np.transpose(rotation_matrices, (2, 0, 1))  # Shape [N, 2, 2]

        # Apply rotation matrices to pos_x, pos_y
        rotated_box_positions = np.einsum('ijk,ik->ij', rotation_matrices, box_coords_duplicated)

        # then apply transformation
        transformed_box_positions = rotated_box_positions + state_transitions[:, :2].repeat(4, 0)
        transformed_box_positions = transformed_box_positions.reshape(-1, 4, 2)

        err = np.linalg.norm(transformed_box_positions - transformed_box_positions[int(rand_idx):int(rand_idx+1)], axis=-1).mean(1)
        err_below_threshold = err < eps
        state_transitions = state_transitions[~err_below_threshold]

    return V


def transform_box_corners_from_vocab(box_coords, V):
    """ Transforms box corner coordinates using a vocabulary of transformations."""
    # box_corners: [T, 4, 2]
    # V: [384, 3]
    # returns: transformed_box_corners: [A, 384, 4, 2]
    
    vocab_size = len(V)
    T = box_coords.shape[0]
    box_coords_duplicated = np.tile(box_coords[:, None, :, :], (1, vocab_size, 1, 1))
    box_coords_duplicated = box_coords_duplicated.reshape(-1, 2)
    V_duplicated = np.tile(V[None, :, None, :], (T, 1, 4, 1))

    rotations = V_duplicated[:,:,:,2].reshape(-1)
    cos_theta = np.cos(rotations)
    sin_theta = np.sin(rotations)

    # Create rotation matrices
    rotation_matrices = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])  # Shape [2, 2, N]
    rotation_matrices = np.transpose(rotation_matrices, (2, 0, 1))  # Shape [N, 2, 2]

    # Apply rotation matrices to pos_x, pos_y
    rotated_box_positions = np.einsum('ijk,ik->ij', rotation_matrices, box_coords_duplicated)

    # then apply transformation
    transformed_box_positions = rotated_box_positions + V_duplicated[:,:,:,:2].reshape(-1, 2)
    transformed_box_positions = transformed_box_positions.reshape(-1, vocab_size, 4, 2)

    return transformed_box_positions


def get_local_state_transition(current_state, next_state):
    """ Computes the relative motion that takes you from current_state 
    to next_state expressed in the local coordinates system of current_state."""
    diff_pos_all = next_state[:, :2] - current_state[:, :2]
    diff_head_all = normalize_angle(next_state[:, 2:] - current_state[:, 2:])
        
    diff_pos_all_reshaped = diff_pos_all.reshape(-1, 2)
    # apply negative of rotation of src state
    rotations_reshaped = -1 * current_state[:, 2].reshape(-1)
    cos_theta = np.cos(rotations_reshaped)
    sin_theta = np.sin(rotations_reshaped)
    rotation_matrices = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])  # Shape [2, 2, N]
    rotation_matrices = np.transpose(rotation_matrices, (2, 0, 1))  # Shape [N, 2, 2]
    
    rotated_diff_pos_all_reshaped = np.einsum('ijk,ik->ij', rotation_matrices, diff_pos_all_reshaped)
    diff_head_all_reshaped = diff_head_all.reshape(-1, 1)

    state_transitions = np.concatenate([rotated_diff_pos_all_reshaped, diff_head_all_reshaped], axis=-1)
    
    return state_transitions


def transform_box_corners_from_local_state(box_coords, local_state_transitions):
    """ Transforms box corner coordinates using local state transitions."""
    # box_coords: [A, 4, 2]
    # local_state_transitions: [A, 3]
    # returns: transformed_box_positions: [A, 4, 2]
    
    local_state_transitions_duplicated = np.tile(local_state_transitions[:, None, :], (1, 4, 1))
    box_coords_duplicated = box_coords.reshape(-1, 2)

    rotations = local_state_transitions_duplicated[:,:,2].reshape(-1)
    cos_theta = np.cos(rotations)
    sin_theta = np.sin(rotations)

    # Create rotation matrices
    rotation_matrices = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])  # Shape [2, 2, N]
    rotation_matrices = np.transpose(rotation_matrices, (2, 0, 1))  # Shape [N, 2, 2]

    # Apply rotation matrices to pos_x, pos_y
    rotated_box_positions = np.einsum('ijk,ik->ij', rotation_matrices, box_coords_duplicated)

    # then apply transformation
    transformed_box_positions = rotated_box_positions + local_state_transitions_duplicated[:,:,:2].reshape(-1, 2)
    transformed_box_positions = transformed_box_positions.reshape(-1, 4, 2)

    return transformed_box_positions


def get_global_next_state(global_states, local_transitions):
    """ Computes the next global states given current global states and local transitions."""
    # global_states: [A, 3]
    # local_transitions: [A, 3] 
    # returns: next state in global frame: [A, 3]
    
    # Extract components
    x, y, heading = global_states[:, 0], global_states[:, 1], global_states[:, 2]
    dx_local, dy_local, d_heading = local_transitions[:, 0], local_transitions[:, 1], local_transitions[:, 2]

    # Compute the rotation matrix components
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)

    # Rotate local transitions to global frame (apply transition matrix)
    dx_global = cos_heading * dx_local - sin_heading * dy_local
    dy_global = sin_heading * dx_local + cos_heading * dy_local

    # Apply the transitions
    x_new = x + dx_global
    y_new = y + dy_global
    heading_new = normalize_angle(heading + d_heading)

    # Combine into resulting global states
    next_global_states = np.stack([x_new, y_new, heading_new], axis=-1)

    return next_global_states
