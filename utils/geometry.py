import numpy as np

def rotate_and_normalize_angles(current_angles, rotation_angle):
    """ Rotates angles by a given rotation angle and normalizes them to be between -pi and pi."""
    # Calculate new angles
    new_angles = current_angles + rotation_angle
    # Normalize the angles to be between -pi and pi
    normalized_angles = (new_angles + np.pi) % (2 * np.pi) - np.pi
    return normalized_angles

def normalize_angle(angle):
    """ Normalizes angle to be between -pi and pi."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def dot_product_2d(a, b):
    """Computes the dot product of 2d vectors."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def cross_product_2d(a, b):
    """Computes the signed magnitude of cross product of 2d vectors."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def make_2d_rotation_matrix(angle_in_radians):
    """ Makes rotation matrix to rotate point in x-y plane counterclockwise by angle_in_radians.
    """
    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                        [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

def apply_se2_transform(coordinates, translation, yaw):
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    """
    coordinates = coordinates - translation
    
    transform = make_2d_rotation_matrix(angle_in_radians=yaw)
    if len(coordinates.shape) > 2:
        coord_shape = coordinates.shape
        return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
    return np.dot(transform, coordinates.T).T[:, :2]

def radians_to_degrees(radians):
    """ Converts radians to degrees."""
    degrees = radians * (180 / 3.141592653589793)
    return degrees

def normalize_lanes_and_agents(agents, lanes, normalize_dict, dataset):
    """ Normalize lanes and agents to coordinate frame defined by the provided normalization dictionary."""
    offset = np.pi / 2 if dataset == 'waymo' else 0
    angle_of_rotation = offset + np.sign(-normalize_dict['yaw']) * np.abs(normalize_dict['yaw'])
    translation = normalize_dict['center'][None, None, :]

    agents_normalized = np.zeros_like(agents)
    agents_normalized[:, :, :2] = apply_se2_transform(
        coordinates=agents[:, :, :2],
        translation=translation,
        yaw=angle_of_rotation)
    
    cos_theta = agents[:, :, 3]
    sin_theta = agents[:, :, 4]
    theta = np.arctan2(sin_theta, cos_theta)
    theta_normalized = rotate_and_normalize_angles(theta, angle_of_rotation.reshape(1, 1))

    agents_normalized[:, :, 2] = agents[:, :, 2]  # keep speed the same
    agents_normalized[:, :, 3] = np.cos(theta_normalized)
    agents_normalized[:, :, 4] = np.sin(theta_normalized)
    agents_normalized[:, :, 5:] = agents[:, :, 5:]  # remaining attributes are se2 invariant

    lanes_normalized = apply_se2_transform(
        coordinates=lanes,
        translation=translation,
        yaw=angle_of_rotation)
    
    return np.squeeze(agents_normalized, axis=1), lanes_normalized