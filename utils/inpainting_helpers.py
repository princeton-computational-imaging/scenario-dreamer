import torch
import numpy as np
import copy
import networkx as nx
import random
from utils.lane_graph_helpers import resample_polyline
from utils.geometry import normalize_lanes_and_agents
from utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite
from utils.torch_helpers import from_numpy
from utils.data_helpers import normalize_scene
from cfgs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN, PARTITIONED


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
    lane_graph_succ = lane_graph_adj_succ.reshape(-1)
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
