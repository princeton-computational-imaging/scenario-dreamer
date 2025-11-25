import numpy as np
from scipy.interpolate import interp1d
from utils.geometry import normalize_angle

def calculate_progress(path):
    """
    Calculate the cumulative progress of a given path.
    :param path: a path consisting of StateSE2 as waypoints
    :return: a cumulative list of progress
    """
    x_position = path[:, 0]
    y_position = path[:, 1]
    x_diff = np.diff(x_position)
    y_diff = np.diff(y_position)
    points_diff = np.concatenate(([x_diff], [y_diff]), axis=0, dtype=np.float64)
    progress_diff = np.append(0.0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff, dtype=np.float64) 

def get_path_length(path):
    """Calculate the length of a given path."""
    return calculate_progress(path)[-1]

def interpolate_path(distances, length, progress, states_se2_array, as_array=False):
    """
    Calculates (x,y,θ) for a given distance along the path.
    :param distances: list of array of distance values
    :param as_array: whether to return in array representation, defaults to False
    :return: array of StateSE2 class or (x,y,θ) values
    """
    clipped_distances = np.clip(distances, 1e-5, length)
    
    _interpolator = interp1d(progress, states_se2_array, axis=0)
    interpolated_se2_array = _interpolator(clipped_distances)
    interpolated_se2_array[..., 2] = normalize_angle(interpolated_se2_array[..., 2])
    interpolated_se2_array[np.isnan(interpolated_se2_array)] = 0.0

    if as_array:
        return interpolated_se2_array
    
def coords_in_frame(coords, frame):
    """
    Checks which coordinates are within the given 2D frame extend.
    :param coords: coordinate array in numpy (x,y) in last axis
    :param frame: tuple of frame extend in meter
    :return: numpy array of boolean's
    """
    assert coords.shape[-1] == 2, "Coordinate array must have last dim size of 2 (ie. x,y)"
    width, height = frame

    within_width = np.logical_and(-width / 2 <= coords[..., 0], coords[..., 0] <= width / 2)
    within_height = np.logical_and(-height / 2 <= coords[..., 1], coords[..., 1] <= height / 2)

    return np.logical_and(within_width, within_height)

def find_consecutive_true_indices(mask):
    """
    Helper function for line preprocessing.
    For example, lines might exceed or return into frame.
    Find regions in mask where line is consecutively in frame (ie. to split line)

    :param mask: 1D numpy array of booleans
    :return: List of int32 arrays, where mask is consecutively true.
    """

    padded_mask = np.pad(np.asarray(mask), (1, 1), "constant", constant_values=False)

    changes = np.diff(padded_mask.astype(int))
    starts = np.where(changes == 1)[0]  # indices of False -> True
    ends = np.where(changes == -1)[0]  # indices of True -> False

    return [np.arange(start, end) for start, end in zip(starts, ends)]


def pixel_in_frame(pixel, pixel_frame):
    """
    Checks if pixels indices are within the image.
    :param pixel: pixel indices as numpy array
    :param pixel_frame: tuple of raster width and height
    :return: numpy array of boolean's
    """
    assert pixel.shape[-1] == 2, "Coordinate array must have last dim size of 2 (ie. x,y)"
    pixel_width, pixel_height = pixel_frame

    within_width = np.logical_and(0 <= pixel[..., 0], pixel[..., 0] < pixel_width)
    within_height = np.logical_and(0 <= pixel[..., 1], pixel[..., 1] < pixel_height)

    return np.logical_and(within_width, within_height)


def coords_to_pixel(coords, frame, pixel_size):
    """
    Converts ego-centric coordinates into pixel coordinates (ie. indices)
    :param coords: coordinate array in numpy (x,y) in last axis
    :param frame: tuple of frame extend in meter
    :param pixel_size: size of a pixel
    :return: indices of pixel coordinates
    """
    assert coords.shape[-1] == 2

    width, height = frame
    pixel_width, pixel_height = int(width / pixel_size), int(height / pixel_size)
    pixel_center = np.array([[pixel_width / 2.0, pixel_height / 2.0]])
    coords_idcs = (coords / pixel_size) + pixel_center

    return coords_idcs.astype(np.int32)