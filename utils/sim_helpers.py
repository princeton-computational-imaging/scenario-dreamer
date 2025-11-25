import numpy as np

from utils.collision_helpers import batched_collision_checker
from utils.lane_graph_helpers import resample_polyline_every
from utils.metrics_helpers import get_lane_length
from utils.geometry import apply_se2_transform

def ego_completed_route(ego_state, route, dist_threshold=2.0):
    """ Check if ego vehicle has completed the route based on distance threshold."""
    last_pos_route = route[-1]
    if np.linalg.norm(last_pos_route - ego_state) < dist_threshold:
        return True 
    else:
        return False


def ego_collided(ego_state, agent_states):
    """ Check if ego vehicle has collided with any agents."""
    agent_exists = agent_states[:, -1] == 1
    
    # [pos_x, pos_y, heading, length, width]
    ego_state_reshaped = ego_state[None, None][:, :, [0,1,4,5,6]]
    agent_states_reshaped = agent_states[agent_exists, None][:, :, [0,1,4,5,6]]
    collision = batched_collision_checker(
        ego_state_reshaped, 
        agent_states_reshaped
    )[:, 0] == 1
    return np.any(collision)


def ego_off_route(ego_state, route, off_route_threshold=5.0):
    """ Check if ego vehicle is off the route based on distance threshold."""
    if len(route) == 1:
        route_dense = route 
    else:
        route_dense = resample_polyline_every(route, 0.1)
    dist_to_route = np.min(np.linalg.norm(route_dense - ego_state[None, :], axis=-1))
    
    return dist_to_route > off_route_threshold


def ego_progress(ego_state, route):
    """ Compute the progress of the ego vehicle along the route."""
    if len(route) == 1:
        route_dense = route 
    else:
        route_dense = resample_polyline_every(route, 0.1)
    end_index = np.argmin(np.linalg.norm(route_dense - ego_state[None, :], axis=-1))

    return get_lane_length(route_dense[:end_index+1])


def normalize_route(route, normalize_dict, offset=np.pi/2):
    """ Normalize route coordinates based on normalization parameters."""
    yaw = normalize_dict['yaw']
    translation = normalize_dict['center']
    
    angle_of_rotation = offset + np.sign(-yaw) * np.abs(yaw)
    translation = translation[np.newaxis, np.newaxis, :]
    
    route = apply_se2_transform(coordinates=route[:, None],
                                translation=translation,
                                yaw=angle_of_rotation)
    return route[:, 0]
