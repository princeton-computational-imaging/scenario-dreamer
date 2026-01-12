import numpy as np
import networkx as nx

from utils.collision_helpers import batched_collision_checker
from utils.lane_graph_helpers import resample_polyline_every, resample_polyline, find_closest_lane
from utils.metrics_helpers import get_lane_length
from utils.geometry import apply_se2_transform

def ego_completed_route(ego_state, route, dist_threshold=2.0):
    """ Check if ego vehicle has completed the route based on distance threshold."""
    last_pos_route = route[-1]
    if np.linalg.norm(last_pos_route - ego_state) < dist_threshold:
        return True 
    else:
        return False


def ego_collided(ego_state, agent_states, agent_scale=1.0):
    """ Check if ego vehicle has collided with any agents."""
    agent_exists = agent_states[:, -1] == 1
    
    # [pos_x, pos_y, heading, length, width]
    ego_state_reshaped = ego_state[None, None][:, :, [0,1,4,5,6]]
    agent_states_reshaped = agent_states[agent_exists, None][:, :, [0,1,4,5,6]]
    
    agent_states_reshaped[:, :, 3] *= agent_scale
    agent_states_reshaped[:, :, 4] *= agent_scale
    ego_state_reshaped[:, :, 3] *= agent_scale
    ego_state_reshaped[:, :, 4] *= agent_scale
    
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


def get_ego_route(compact_lane_graph, lanes, ego_trajectory, dist_threshold=10):
    """ Get the ego route from the lane graph."""
    suc_pairs = compact_lane_graph['suc_pairs']
    num_lanes = len(lanes)

    start_position = ego_trajectory[0, :2]
    invalid = np.where(ego_trajectory[:, -1] == 0)[0]
    t_end = invalid[0] if len(invalid) else len(ego_trajectory)
    
    if t_end == 0:
        return None
    
    end_position = ego_trajectory[t_end - 1, :2]
    ego_trajectory = ego_trajectory[:t_end, :2]
    
    dist_to_lanes = np.linalg.norm(
        lanes[:, None] - ego_trajectory[None, :, None, :2]
        , axis=-1).min(-1).min(-1)
    lane_mask = dist_to_lanes < dist_threshold
    lanes = lanes[lane_mask]

    if len(lanes) == 0:
        return None

    old_to_new_id = {}
    new_to_old_id = {}
    count = 0
    for i in range(num_lanes):
        if lane_mask[i] == 0:
            continue 
        
        old_to_new_id[i] = count 
        new_to_old_id[count] = i 
        count += 1 
    
    num_lanes = count

    A = np.zeros((num_lanes, num_lanes))
    for lane_id in suc_pairs:
        if lane_id not in old_to_new_id:
            continue
        for suc_lane_id in suc_pairs[lane_id]:
            if suc_lane_id not in old_to_new_id:
                continue
            A[old_to_new_id[lane_id], old_to_new_id[suc_lane_id]] = 1
    G = nx.DiGraph(incoming_graph_data=A)

    start_lane = find_closest_lane(lanes, start_position)
    end_lane = find_closest_lane(lanes, end_position)

    try:
        shortest_paths = nx.all_simple_paths(G, start_lane, end_lane)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    
    candidate_paths = []
    for path in shortest_paths:
        path_lanes = []
        for i, lane_id in enumerate(path):
            if i == 0 and len(path) == 1:
                end_idx = np.argmin(
                    np.linalg.norm(
                        lanes[lane_id] - end_position[None, :]
                        , axis=-1))
                start_idx = np.argmin(
                    np.linalg.norm(
                        lanes[lane_id] - start_position[None, :]
                        , axis=-1))
                if start_idx == end_idx:
                    end_idx += 1
                path_lanes.append(lanes[lane_id, start_idx:end_idx+1])
            elif i == 0:
                start_idx = np.argmin(
                    np.linalg.norm(
                        lanes[lane_id] - start_position[None, :]
                        , axis=-1))
                path_lanes.append(lanes[lane_id, start_idx:])
            elif i == len(path) - 1:
                end_idx = np.argmin(
                    np.linalg.norm(
                        lanes[lane_id] - end_position[None, :]
                        , axis=-1))
                path_lanes.append(lanes[lane_id, :end_idx+1])
            else:
                path_lanes.append(lanes[lane_id])

        path_lanes = np.concatenate(path_lanes, axis=0)
        if len(path_lanes) > 0:
            path_lanes = resample_polyline(
                path_lanes, 
                num_points=100)
            candidate_paths.append(path_lanes[None, :])
    
    if len(candidate_paths) == 0:
        return None

    candidate_paths = np.concatenate(
        candidate_paths, axis=0)[:, None, :, :] 
    ego_trajectory = ego_trajectory[None, :, None, :]   
    ego_path_dists = np.linalg.norm(
        candidate_paths - ego_trajectory, axis=-1)
    best_candidate_path = np.argmin(
        ego_path_dists.min(-1).sum(1))

    ego_route = candidate_paths[best_candidate_path, 0]
    ego_route = resample_polyline_every(ego_route, every=1)

    return ego_route
