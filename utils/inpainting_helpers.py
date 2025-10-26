import torch
import numpy as np
import copy
import networkx as nx
import random
from utils.lane_graph_helpers import resample_polyline
from utils.geometry import apply_se2_transform, rotate_and_normalize_angles, normalize_angle
from utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite
from utils.torch_helpers import from_numpy
from utils.metrics_helpers import get_lane_graph_waymo, get_lane_graph_nuplan
from utils.data_helpers import normalize_scene
from cfgs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN, PARTITIONED


def estimate_heading(positions):
    """ Compute heading at the start and end of a sequence of positions."""
    # positions: numpy array of shape (20, 2) representing (x, y) positions

    # Estimate heading for the first point
    diff_first = positions[1] - positions[0]
    heading_first = np.arctan2(diff_first[1], diff_first[0])

    # Estimate heading for the last point
    diff_last = positions[-1] - positions[-2]
    heading_last = np.arctan2(diff_last[1], diff_last[0])

    return heading_first, heading_last


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


def normalize_and_crop_scene(cond_d, new_d, normalize_dict, cfg, dataset_name, num_upsample_points=1000):
    """ Normalize and crop lanes and agents as preprocessing step for inpainting."""
    lanes = cond_d['road_points']
    agents = cond_d['agent_states']

    if dataset_name == 'waymo':
        PARTITION_IDX = 1  # y-axis partition for Waymo 
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_WAYMO
    else:
        PARTITION_IDX = 0 # x-axis partition for Nuplan
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_NUPLAN

    # upsample lanes, so that cropping is more accurate
    lanes_upsampled = []
    for lane in lanes:
        lanes_upsampled.append(resample_polyline(lane, num_points=num_upsample_points))
    lanes = np.array(lanes_upsampled)

    # normalize lane and agents to endpoint of route, and transport ego to new scene center
    old_ego_state = copy.deepcopy(agents[0])
    agents, lanes = normalize_lanes_and_agents(agents[:, None], lanes, normalize_dict, dataset=dataset_name)
    # first agent is ego, set its position to origin
    agents[0] = np.array([
        0.0,  # x
        0.0,  # y
        old_ego_state[2],  # speed
        0.0 if dataset_name == 'waymo' else 1.0,  # cos(yaw)
        1.0 if dataset_name == 'waymo' else 0.0,  # sin(yaw)
    ] + list(old_ego_state[5:]))

    agent_types = cond_d['agent_types']
    if dataset_name == 'nuplan':
        lane_types = cond_d['lane_types']
    map_id = int(cond_d['map_id'])
    num_lanes = len(lanes)

    lane_point_dists_x = np.abs(lanes[:, :, 0])
    lane_point_dists_y = np.abs(lanes[:, :, 1])
    lane_points_mask_within_fov = np.logical_and(
        lane_point_dists_x < cfg.fov / 2,
        lane_point_dists_y < cfg.fov / 2
    )

    lane_points_mask_before_partition = lanes[:, :, PARTITION_IDX] <= 0
    lane_points_mask_bp_and_wf = np.logical_and(
        lane_points_mask_within_fov,
        lane_points_mask_before_partition
    )

    # lanes before partition (bp) and within fov (wf)
    lanes_mask_bp_and_wf = np.any(
        lane_points_mask_bp_and_wf, axis=1)
    lane_ids_bp_and_wf = np.where(lanes_mask_bp_and_wf)[0]
    lane_ids_others = np.setdiff1d(np.arange(len(lanes)), lane_ids_bp_and_wf)

    lanes = lanes[lanes_mask_bp_and_wf]
    lane_points_mask = lane_points_mask_bp_and_wf[lanes_mask_bp_and_wf]

    # resample lanes within new FOV
    resampled_lanes = []
    for mask, lane in zip(lane_points_mask, lanes):
        valid_points = lane[mask]
        resampled_lane = resample_polyline(valid_points, num_points=20)
        resampled_lanes.append(resampled_lane)
    lanes_bp_and_wf = np.array(resampled_lanes)
    if dataset_name == 'nuplan':
        lane_types_bp_and_wf = lane_types[lanes_mask_bp_and_wf]

    assert len(lane_ids_bp_and_wf) == len(lanes_bp_and_wf), "Mismatch in lane filtering."

    agent_dists_x = np.abs(agents[:, 0])
    agent_dists_y = np.abs(agents[:, 1])
    agents_mask_within_fov = np.logical_and(
        agent_dists_x < cfg.fov / 2,
        agent_dists_y < cfg.fov / 2
    )
    agents_mask_before_partition = agents[:, PARTITION_IDX] <= 0
    agents_mask_bp_and_wf = np.logical_and(
        agents_mask_within_fov,
        agents_mask_before_partition
    )
    agents_bp_and_wf = agents[agents_mask_bp_and_wf]
    agent_types_bp_and_wf = agent_types[agents_mask_bp_and_wf]

    assert len(lanes_bp_and_wf) > 0, "Number of lanes before partition and within FOV should be greater than 0."
    assert len(agents_bp_and_wf) > 0, "Number of agents before partition and within FOV should be greater than 0."

    lane_graph_adj_pre = cond_d['road_connection_types'][:, LANE_CONNECTION_TYPES['pred']].reshape(num_lanes, num_lanes)
    # remove rows and columns corresponding to removed lanes
    if len(lane_ids_others) > 0:
        lane_graph_adj_pre = np.delete(lane_graph_adj_pre, lane_ids_others, axis=0)
        lane_graph_adj_pre = np.delete(lane_graph_adj_pre, lane_ids_others, axis=1)

    lane_graph_adj_succ = cond_d['road_connection_types'][:, LANE_CONNECTION_TYPES['succ']].reshape(num_lanes, num_lanes)
    if len(lane_ids_others) > 0:
        lane_graph_adj_succ = np.delete(lane_graph_adj_succ, lane_ids_others, axis=0)
        lane_graph_adj_succ = np.delete(lane_graph_adj_succ, lane_ids_others, axis=1)
    
    # we do not model left/right connections for nuplan
    if dataset_name == 'waymo':
        lane_graph_adj_left = cond_d['road_connection_types'][:, LANE_CONNECTION_TYPES['left']].reshape(num_lanes, num_lanes)
        if len(lane_ids_others) > 0:
            lane_graph_adj_left = np.delete(lane_graph_adj_left, lane_ids_others, axis=0)
            lane_graph_adj_left = np.delete(lane_graph_adj_left, lane_ids_others, axis=1)
        
        lane_graph_adj_right = cond_d['road_connection_types'][:, LANE_CONNECTION_TYPES['right']].reshape(num_lanes, num_lanes)
        if len(lane_ids_others) > 0:
            lane_graph_adj_right = np.delete(lane_graph_adj_right, lane_ids_others, axis=0)
            lane_graph_adj_right = np.delete(lane_graph_adj_right, lane_ids_others, axis=1)
    
    lane_graph_pre = lane_graph_adj_pre.reshape(-1)
    lane_graph_succ = lane_graph_adj_succ.transpose(1, 0).reshape(-1)
    lane_graph_self = np.eye(lane_graph_adj_pre.shape[0]).reshape(-1)
    if dataset_name == 'waymo':
        lane_graph_left = lane_graph_adj_left.reshape(-1)
        lane_graph_right = lane_graph_adj_right.reshape(-1)

    road_connection_types_bp_and_wf = np.zeros(len(lane_graph_pre)).astype(int)
    road_connection_types_bp_and_wf[lane_graph_pre == 1] = LANE_CONNECTION_TYPES['pred']
    road_connection_types_bp_and_wf[lane_graph_succ == 1] = LANE_CONNECTION_TYPES['succ']
    if dataset_name == 'waymo':
        road_connection_types_bp_and_wf[lane_graph_left == 1] = LANE_CONNECTION_TYPES['left']
        road_connection_types_bp_and_wf[lane_graph_right == 1] = LANE_CONNECTION_TYPES['right']
    # edge case for 1 lane
    if len(lane_graph_pre) == 1:
        road_connection_types_bp_and_wf[0] = LANE_CONNECTION_TYPES['self']
    else:
        road_connection_types_bp_and_wf[lane_graph_self == 1] = LANE_CONNECTION_TYPES['self']
    road_connection_types_bp_and_wf = np.eye(6 if dataset_name == 'waymo' else 4)[road_connection_types_bp_and_wf]

    num_lanes = len(lanes_bp_and_wf)
    num_agents = len(agents_bp_and_wf)

    agents_bp_and_wf, lanes_bp_and_wf = normalize_scene(
        agents_bp_and_wf, 
        lanes_bp_and_wf,
        fov=cfg.fov,
        min_speed=cfg.min_speed,
        max_speed=cfg.max_speed,
        min_length=cfg.min_length,
        max_length=cfg.max_length,
        min_width=cfg.min_width,
        max_width=cfg.max_width,
        min_lane_x=cfg.min_lane_x,
        min_lane_y=cfg.min_lane_y,
        max_lane_x=cfg.max_lane_x,
        max_lane_y=cfg.max_lane_y
    )

    # build data dictionary for autoencoder processing
    # as we need agent/lane latents for existing half of the scene
    new_d['map_id'] = map_id
    new_d['num_lanes'] = num_lanes
    new_d['num_agents'] = num_agents
    new_d['lg_type'] = PARTITIONED
    new_d['agent'].x = from_numpy(agents_bp_and_wf)
    new_d['agent'].type = from_numpy(agent_types_bp_and_wf)
    new_d['lane'].x = from_numpy(lanes_bp_and_wf)
    new_d['lane'].partition_mask = torch.ones(num_lanes).bool()  # everything is before the partition
    new_d['lane'].ids = from_numpy(lane_ids_bp_and_wf)
    # only nuplan has lane types (lane/green light/red light)
    if dataset_name == 'nuplan':
        new_d['lane'].type = from_numpy(lane_types_bp_and_wf)
    
    new_d['lane', 'to', 'lane'].edge_index = get_edge_index_complete_graph(num_lanes)
    new_d['lane', 'to', 'lane'].type = from_numpy(road_connection_types_bp_and_wf)
    new_d['agent', 'to', 'agent'].edge_index = get_edge_index_complete_graph(num_agents)
    new_d['lane', 'to', 'agent'].edge_index = get_edge_index_bipartite(num_lanes, num_agents)
    new_d['lane', 'to', 'lane'].encoder_mask = torch.ones(new_d['lane', 'to', 'lane'].edge_index.shape[1]).bool()
    new_d['lane', 'to', 'agent'].encoder_mask = torch.ones(new_d['lane', 'to', 'agent'].edge_index.shape[1]).bool()
    new_d['agent', 'to', 'agent'].encoder_mask = torch.ones(new_d['agent', 'to', 'agent'].edge_index.shape[1]).bool()

    return new_d


def sample_num_lanes_agents_inpainting(
    lane_dis,
    map_id,
    num_cond_lanes,
    num_cond_agents,
    max_num_lanes,
    inpainting_prob_matrix,
):
    """ Sample the number of lanes and agents for inpainting. Returns num_inpainted_lanes+num_cond_lanes, num_inpainted_agents+num_cond_agents."""
    batch_size = lane_dis.shape[0]
    # sample from learned distribution
    extra_num_lanes = torch.multinomial(lane_dis, 1).squeeze(-1)
    num_lanes_all = torch.clip(num_cond_lanes + extra_num_lanes, 1, max_num_lanes)
        
    # sample from conditional distribution (num_agents_after_partition | num_lanes, num_agents_before_partition)
    prob_matrix_batch = inpainting_prob_matrix[map_id]
    prob_agent_batch = torch.zeros_like(prob_matrix_batch[:, 0])
    for i in range(batch_size):
        prob_agent_batch[i, :prob_matrix_batch.shape[2] - num_cond_agents[i]] = prob_matrix_batch[i, num_lanes_all[i], num_cond_agents[i]:]
        if prob_agent_batch[i].sum() == 0:
            prob_agent_batch[i, 1] = 1
        else:
            prob_agent_batch[i] /= prob_agent_batch[i].sum()
    
    extra_num_agents = torch.multinomial(prob_agent_batch, 1).squeeze(-1)
    num_agents_all = num_cond_agents + extra_num_agents

    return num_lanes_all, num_agents_all


def sample_route(d, dataset, heading_tolerance=np.pi/3, num_points_in_route=1000):
    """ Sample a valid route for the ego vehicle in the scene."""
    
    if dataset == 'waymo':
        G, lanes = get_lane_graph_waymo(d)
    else:
        G, lanes = get_lane_graph_nuplan(d)

    start_lane = np.linalg.norm(lanes, axis=-1).min(1).argmin()

    # ensure lane heading is sufficiently aligned with ego heading
    start_idx = np.argmin(np.linalg.norm(lanes[start_lane], axis=-1))
    if start_idx == lanes.shape[1] - 1:
        diff = lanes[start_lane][start_idx] - lanes[start_lane][start_idx - 1]
    else:
        diff = lanes[start_lane][start_idx + 1] - lanes[start_lane][start_idx]
    lane_heading = np.arctan2(diff[1], diff[0])
    
    offset = np.pi / 2 if dataset == 'waymo' else 0
    if np.abs(normalize_angle(lane_heading-offset)) >= heading_tolerance:
        return None, False

    # get all simple paths from start_lane to all other lanes
    paths = [[start_lane]]
    for target in G.nodes:
        if target != start_lane:
            paths.extend(nx.all_simple_paths(G, start_lane, target))
    
    valid_paths = []
    for path in paths:
        last_lane = path[-1]
        if not (_near_border(lanes[last_lane, -1]) and _valid_route_end(last_lane, lanes[last_lane])):
            continue 
        valid_paths.append(path)

    if len(valid_paths) == 0:
        return None, False
    
    # randomly select valid route
    random.shuffle(valid_paths)
    route_as_lane_ids = valid_paths[0]

    # construct route as polyline
    route_lanes = []
    for i, lane_id in enumerate(route_as_lane_ids):
        if i == 0 and len(route_as_lane_ids) == 1:
            end_idx = 20
            start_idx = np.argmin(np.linalg.norm(lanes[lane_id], axis=-1))
            if start_idx == end_idx:
                end_idx += 1
            route_lanes.append(lanes[lane_id, start_idx:end_idx])
        elif i == 0:
            start_idx = np.argmin(np.linalg.norm(lanes[lane_id], axis=-1))
            route_lanes.append(lanes[lane_id, start_idx:])
        else:
            route_lanes.append(lanes[lane_id])
    
    route_lanes = np.concatenate(route_lanes, axis=0)
    assert len(route_lanes) > 0, "Route lanes should have more than 0 points."
    route_lanes = resample_polyline(route_lanes, num_points=num_points_in_route)
    return route_lanes, True


def get_default_route_center_yaw(dataset):
    """ Get default route center and yaw for the dataset if no valid route is found."""
    if dataset == 'waymo':
        return np.array([0, 32]), np.pi/2
    else:
        return np.array([32, 0]), 0


def _near_border(pos, fov=64, threshold=1):
    """ Check if position is near border of FOV."""
    if np.abs(np.abs(pos[0]) - fov/2) < threshold or np.abs(np.abs(pos[1]) - fov/2) < threshold:
        return True 
    return False


def _valid_route_end(lane_id, lane, fov=64, border_threshold=1, heading_threshold=5*np.pi/180):
    """ Check if route end is valid (heading aligned with corresponding border)."""
    _, last_heading = estimate_heading(lane)

    last_pos = lane[-1]
    # route end near positive y axis
    if np.abs(last_pos[1] - fov/2) < border_threshold:
        target_angle = np.pi/2
    # route end near negative y axis
    elif np.abs(last_pos[1] - -fov/2) < border_threshold:
        target_angle = -np.pi/2
    # route end near positive x axis
    elif np.abs(last_pos[0] - fov/2) < border_threshold:
        target_angle = 0
    # route end near negative x axis
    elif np.abs(last_pos[0] - -fov/2) < border_threshold:
        target_angle = np.pi

    differences = last_heading - target_angle
    normalized_differences = normalize_angle(differences)
    closest_difference = np.abs(normalized_differences)

    return closest_difference <= heading_threshold
