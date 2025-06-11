import numpy as np

def rotate_and_normalize_angles(current_angles, rotation_angle):
    # Calculate new angles
    new_angles = current_angles + rotation_angle
    # Normalize the angles to be between -pi and pi
    normalized_angles = (new_angles + np.pi) % (2 * np.pi) - np.pi
    return normalized_angles

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
    degrees = radians * (180 / 3.141592653589793)
    return degrees