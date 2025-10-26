import numpy as np
import torch
np.set_printoptions(suppress=True)
from utils.data_container import get_batches, get_features
from typing import Tuple
from cfgs.config import PARTITIONED
import os
import pickle

def get_object_type_onehot_waymo(agent_type):
    """Return the one-hot NumPy vector encoding of an agent type."""
    agent_types = {"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
    return np.eye(len(agent_types))[agent_types[agent_type]]

def get_lane_connection_type_onehot_waymo(lane_connection_type):
    """Return the one-hot NumPy vector encoding of a lane-connection type."""
    lane_connection_types = {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}
    return np.eye(len(lane_connection_types))[lane_connection_types[lane_connection_type]]

def get_lane_connection_type_onehot_nuplan(lane_connection_type):
    """Converts a lane connection type to a one-hot encoded vector."""
    lane_connection_types = {"none": 0, "pred": 1, "succ": 2, "self": 3}
    return np.eye(len(lane_connection_types))[lane_connection_types[lane_connection_type]]

def get_lane_type_onehot_nuplan(lane_type):
    """Converts a lane type to a one-hot encoded vector."""
    lane_types = {"lane": 0, "green_light": 1, "red_light": 2}
    return np.eye(len(lane_types))[lane_types[lane_type]]

def get_object_type_onehot_nuplan(agent_type):
    """Converts an agent type to a one-hot encoded vector."""
    agent_types = {"vehicle": 0, "pedestrian": 1, "static_object": 2}
    return np.eye(len(agent_types))[agent_types[agent_type]]


def reorder_indices(
        agent_mu: np.ndarray,
        agent_log_var: np.ndarray,
        lane_mu: np.ndarray,
        lane_log_var: np.ndarray,
        edge_index_lane_to_lane: np.ndarray,
        agent_states: np.ndarray,
        road_points: np.ndarray,
        lg_type: int,
        tolerance: float = 0.5 / 32,
        dataset: str = 'waymo'
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
    """Reorder agents and lanes to ensure deterministic ordering. This makes the positional
    encodings more meaningful.

    The routine performs a **hierarchical sort** on non-ego agents and on road
    lanes over the following metrics in the prescribed order: [min_y, min_x, max_y, max_x]

    A *tolerance* is applied so that small positional differences do not change the order.  After sorting, all latent
    representations, state tensors, and graph indices are permuted
    consistently.  The ego agent (index 0) is **never moved**.

    Parameters
    ----------
    agent_mu : np.ndarray
        Mean of the Gaussian latent variables for each agent with shape
        ``(N_agents, latent_dim)``.
    agent_log_var : np.ndarray
        Log-variance of the Gaussian latent variables for each agent with the
        same shape as *agent_mu*.
    lane_mu : np.ndarray
        Mean of the Gaussian latent variables for each lane with shape
        ``(N_lanes, latent_dim)``.
    lane_log_var : np.ndarray
        Log-variance of the Gaussian latent variables for each lane with the
        same shape as *lane_mu*.
    edge_index_lane_to_lane : np.ndarray
        Edge list of the lane-to-lane graph in COO format with shape
        ``(2, E)`` or ``(E, 2)``.  Indices are updated to reflect the new lane
        order.
    agent_states : np.ndarray
        Full state tensor for agents used to derive the sort keys.
    road_points : np.ndarray
        Sampled poly-line points for each lane with shape
        ``(N_lanes, N_points, 2)``.
    lg_type : int
        Scene layout type.  If ``lg_type == 1`` the function marks agents and
        lanes that are south of the horizontal partition (``y <= 0``).
        Otherwise no partitioning mask is applied.
    tolerance : float, optional
        Numerical tolerance (in the same units as coordinates) within which
        metric differences are considered equal.  Defaults to ``0.5 / 32``
        (≈0.0156).
    dataset : str, optional
        Either waymo or nuplan, which determines orientation of scene and therefore recursive ordering

    Returns
    -------
    Tuple[np.ndarray, ...]
        A 7-tuple containing:

        1. **agent_mu_sorted** - permuted *agent_mu* with ego agent first.
        2. **agent_log_var_sorted** - permuted *agent_log_var*.
        3. **lane_mu_sorted** - permuted *lane_mu*.
        4. **lane_log_var_sorted** - permuted *lane_log_var*.
        5. **edge_index_lane_to_lane_new** - updated edge indices.
        6. **agent_partition_mask** - boolean mask of shape ``(N_agents,)``
        indicating agents below the ``y=0`` partition when
        ``lg_type == 1``.
        7. **lane_partition_mask** - boolean mask of shape ``(N_lanes,)``
        indicating lanes below the partition when ``lg_type == 1``.

    Notes
    -----
    • The sorting of agents excludes the ego agent (index ``0``), which is
    re-inserted at the head of every returned tensor.

    • When *road_points* is empty (``shape[0] == 0``) the lane-related outputs
    are returned unchanged.
    """
    
    def hierarchical_sort(values, metrics, tolerance):
        """
        Recursively sorts indices based on a list of metrics and a tolerance.
        """
        indices = np.arange(len(values[metrics[0]]))
        
        def sort_recursive(indices, metric_idx):
            if len(indices) == 0:
                return indices  # Return empty array if no indices to sort
            if metric_idx >= len(metrics):
                return indices
            
            metric = metrics[metric_idx]
            values_metric = values[metric][indices]
            sorted_order = np.argsort(values_metric)
            indices = indices[sorted_order]
            values_metric_sorted = values_metric[sorted_order]
            
            # Group indices where the difference is less than tolerance
            groups = []
            current_group = [indices[0]]
            for i in range(1, len(indices)):
                diff = values_metric_sorted[i] - values_metric_sorted[i - 1]
                if diff < tolerance:
                    current_group.append(indices[i])
                else:
                    # Recursively sort the current group if needed
                    if len(current_group) > 1:
                        current_group = sort_recursive(np.array(current_group), metric_idx + 1).tolist()
                    groups.extend(current_group)
                    current_group = [indices[i]]
            # Handle the last group
            if len(current_group) > 1:
                current_group = sort_recursive(np.array(current_group), metric_idx + 1).tolist()
            groups.extend(current_group)
            return np.array(groups)
        
        return sort_recursive(indices, 0)
    
    if dataset == 'waymo':
        PARTITION_IDX = 1  # y-axis partition for Waymo 
    else:
        PARTITION_IDX = 0 # x-axis partition for Nuplan
    
    # Process Agents (ego is first index)
    non_ego_agent_mu = agent_mu[1:]
    non_ego_agent_log_var = agent_log_var[1:]
    non_ego_agent_states = agent_states[1:]
    
    if non_ego_agent_states.shape[0] > 0:
        # Calculate metrics for agents
        agent_min_y = non_ego_agent_states[:, 1]
        agent_min_x = non_ego_agent_states[:, 0]
        agent_max_y = non_ego_agent_states[:, 1]
        agent_max_x = non_ego_agent_states[:, 0]
        
        agent_values = {
            'min_y': agent_min_y,
            'min_x': agent_min_x,
            'max_y': agent_max_y,
            'max_x': agent_max_x
        }
        
        if dataset == 'waymo':
            agent_metrics = ['min_y', 'min_x', 'max_y', 'max_x']
        else:
            agent_metrics = ['min_x', 'min_y', 'max_x', 'max_y']
        perm = hierarchical_sort(agent_values, agent_metrics, tolerance)
        
        # Reorder non-ego agents
        non_ego_agent_mu = non_ego_agent_mu[perm]
        non_ego_agent_log_var = non_ego_agent_log_var[perm]
        non_ego_agent_states = non_ego_agent_states[perm]
    
    # Concatenate ego agent back
    agent_mu = np.concatenate([agent_mu[:1], non_ego_agent_mu], axis=0)
    agent_log_var = np.concatenate([agent_log_var[:1], non_ego_agent_log_var], axis=0)
    agent_states_sorted = np.concatenate([agent_states[:1], non_ego_agent_states], axis=0)

    # which agents are below the partition
    if lg_type == PARTITIONED:
        agent_partition_mask = agent_states_sorted[:, PARTITION_IDX] <= 0
    else:
        agent_partition_mask = np.zeros_like(agent_states_sorted[:, PARTITION_IDX] <= 0)
    
    if road_points.shape[0] > 0:
        lane_min_y = np.min(road_points[:, :, 1], axis=1)
        lane_min_x = np.min(road_points[:, :, 0], axis=1)
        lane_max_y = np.max(road_points[:, :, 1], axis=1)
        lane_max_x = np.max(road_points[:, :, 0], axis=1)
        
        lane_values = {
            'min_y': lane_min_y,
            'min_x': lane_min_x,
            'max_y': lane_max_y,
            'max_x': lane_max_x
        }
        
        if dataset == 'waymo':
            lane_metrics = ['min_y', 'min_x', 'max_y', 'max_x']
        else:
            lane_metrics = ['min_x', 'min_y', 'max_x', 'max_y']
        lane_perm = hierarchical_sort(lane_values, lane_metrics, tolerance)
        
        # Reorder lanes
        lane_mu = lane_mu[lane_perm]
        lane_log_var = lane_log_var[lane_perm]

        road_points_sorted = road_points[lane_perm]
        # which roads are below the partition
        if lg_type == PARTITIONED:
            lane_partition_mask = road_points_sorted[:, 9, PARTITION_IDX] <= 0
        else:
            lane_partition_mask = np.zeros_like(road_points_sorted[:, 9, PARTITION_IDX] <= 0)
        
        # Update edge indices
        old_index_positions = np.argsort(lane_perm)
        edge_index_lane_to_lane_new = old_index_positions[edge_index_lane_to_lane]
    else:
        edge_index_lane_to_lane_new = edge_index_lane_to_lane  # No change if no lanes
        # no lanes
        lane_partition_mask = road_points[:, 9, PARTITION_IDX] <= 0
    
    return agent_mu, agent_log_var, lane_mu, lane_log_var, edge_index_lane_to_lane_new, agent_partition_mask, lane_partition_mask


def modify_agent_states(agent_states):
    """Canonicalise velocity & heading for neural consumption. All remaining trailing columns (if any) are copied verbatim.

    Parameters
    ----------
    agent_states : np.ndarray
        Float32 array of shape ``(N, D)`` where columns ``2-4`` are
        ``vx``, ``vy``, and ``yaw`` respectively.

    Returns
    -------
    new_agent_states : np.ndarray
        Array with the *same* shape ``(N, D)`` where columns ``2-4``
        have been replaced by ``speed``, ``cosθ``, ``sinθ``.
    """
    new_agent_states = np.zeros_like(agent_states)
    new_agent_states[:, :2] = agent_states[:, :2]
    new_agent_states[:, 5:] = agent_states[:, 5:]
    new_agent_states[:, 2] = np.sqrt(agent_states[:, 2] ** 2 + agent_states[:, 3] ** 2)
    new_agent_states[:, 3] = np.cos(agent_states[:, 4])
    new_agent_states[:, 4] = np.sin(agent_states[:, 4])

    return new_agent_states


def normalize_scene(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max normalise agent and lane features into **[-1, 1]**."""
    
    # pos_x
    agent_states[:, 0] = 2 * ((agent_states[:, 0] - (-1 * fov/2))
                            / fov) - 1
    # pos_y
    agent_states[:, 1] = 2 * ((agent_states[:, 1] - (-1 * fov/2))
                            / fov) - 1
    # speed
    agent_states[:, 2] = 2 * ((agent_states[:, 2] - (min_speed))
                            / (max_speed - min_speed)) - 1
    # length
    agent_states[:, 5] = 2 * ((agent_states[:, 5] - (min_length))
                            / (max_length - min_length)) - 1
    # width
    agent_states[:, 6] = 2 * ((agent_states[:, 6] - (min_width))
                            / (max_width - min_width)) - 1
    
    # road pos_x
    road_points[:, :, 0] = 2 * ((road_points[:, :, 0] - (min_lane_x))
                            / (max_lane_x - min_lane_x)) - 1
    road_points[:, :, 1] = 2 * ((road_points[:, :, 1] - (min_lane_y))
                            / (max_lane_y - min_lane_y)) - 1

    return agent_states, road_points


def unnormalize_scene(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Unnormalize the agent states and lane points from a range of [-1, 1] to their original scale based on the dataset configuration."""
    # pos_x
    agent_states[:, 0] = ((torch.clip(agent_states[:, 0], -1, 1) + 1) / 2) * fov + (-1 * fov/2)
    # pos_y
    agent_states[:, 1] = ((torch.clip(agent_states[:, 1], -1, 1) + 1) / 2) * fov + (-1 * fov/2)
    # speed
    agent_states[:, 2] = ((torch.clip(agent_states[:, 2], -1, 1) + 1) / 2) * (max_speed - min_speed) + min_speed
    # cos_theta
    agent_states[:, 3] = torch.clip(agent_states[:, 3], -1, 1)
    # sin_theta
    agent_states[:, 4] = torch.clip(agent_states[:, 4], -1, 1)
    # length
    agent_states[:, 5] = ((torch.clip(agent_states[:, 5], -1, 1) + 1) / 2) * (max_length - min_length) + min_length
    # width
    agent_states[:, 6] = ((torch.clip(agent_states[:, 6], -1, 1) + 1) / 2) * (max_width - min_width) + min_width

    lower_clip = -1000
    upper_clip = 1000
    
    # lane pos_x
    road_points[:, :, 0] = ((torch.clip(road_points[:, :, 0], lower_clip, upper_clip) + 1) / 2) * (max_lane_x - min_lane_x) + min_lane_x
    # lane pos_y
    road_points[:, :, 1] = ((torch.clip(road_points[:, :, 1], lower_clip, upper_clip) + 1) / 2) * (max_lane_y - min_lane_y) + min_lane_y

    return agent_states, road_points


def randomize_indices(
    agent_states,
    agent_types,
    road_points,
    edge_index_lane_to_lane,
    lane_types=None):
    """Randomly permute non-ego agents and lane order during training."""
    non_ego_agent_states = agent_states[1:]
    non_ego_agent_types = agent_types[1:]

    num_non_ego_agents = len(non_ego_agent_states)
    perm = np.arange(num_non_ego_agents)
    np.random.shuffle(perm)
    non_ego_agent_states = non_ego_agent_states[perm]
    non_ego_agent_types = non_ego_agent_types[perm]

    agent_states = np.concatenate([agent_states[:1], non_ego_agent_states], axis=0)
    agent_types = np.concatenate([agent_types[:1], non_ego_agent_types], axis=0)

    lane_perm = np.arange(len(road_points))
    np.random.shuffle(lane_perm)
    road_points = road_points[lane_perm]
    if lane_types is not None:
        lane_types = lane_types[lane_perm]
    
    old_index_positions = np.argsort(lane_perm)
    edge_index_lane_to_lane_new = old_index_positions[edge_index_lane_to_lane]

    if lane_types is not None:
        return agent_states, agent_types, road_points, lane_types, edge_index_lane_to_lane_new
    else:
        return agent_states, agent_types, road_points, edge_index_lane_to_lane_new
    

def normalize_latents(
        agent_latents, 
        lane_latents,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std):
    """ Normalize the agent and lane latents using the mean and std from the config."""
    agent_latents = (agent_latents - agent_latents_mean) / agent_latents_std
    lane_latents = (lane_latents - lane_latents_mean) / lane_latents_std

    return agent_latents, lane_latents


def unnormalize_latents(
        agent_latents, 
        lane_latents,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std):
    """ Unnormalize the agent and lane latents using the mean and std from the config."""
    agent_latents = agent_latents * agent_latents_std + agent_latents_mean
    lane_latents = lane_latents * lane_latents_std + lane_latents_mean

    return agent_latents, lane_latents


def reparameterize(mu, log_var):
    """ Reparameterization trick to sample from a Gaussian distribution
    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution.
        log_var (torch.Tensor): Log variance of the Gaussian distribution.
    Returns:
        torch.Tensor: Sampled latent variable.
    """
    assert mu.shape == log_var.shape
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def sample_latents(
        data, 
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std,
        normalize=True):
    """ Sample latents from the agent and lane data, and (optionally) normalize them."""
    agent_mu = data['agent'].x
    agent_log_var = data['agent'].log_var 
    agent_latents = reparameterize(agent_mu, agent_log_var)

    lane_mu = data['lane'].x 
    lane_log_var = data['lane'].log_var 
    lane_latents = reparameterize(lane_mu, lane_log_var)

    if normalize:
        agent_latents, lane_latents = normalize_latents(
            agent_latents, 
            lane_latents,
            agent_latents_mean,
            agent_latents_std,
            lane_latents_mean,
            lane_latents_std)
    
    return agent_latents, lane_latents


def convert_batch_to_scenarios(data, batch_idx, cache_dir, cache_samples=False, cache_lane_types=False):
    """ Converts batch output into individual scenarios. Optionally saves scenarios to disk."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    batch_size = data.batch_size
    agent_batch, lane_batch, lane_conn_batch = get_batches(data)
    x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
    
    # move to cpu
    lg_type = data['lg_type'].cpu().numpy()
    map_ids = data['map_id'].cpu().numpy()
    x_agent_states = x_agent_states.cpu().numpy()
    x_agent_types = x_agent_types.cpu().numpy()
    if cache_lane_types:
        x_lane_types = x_lane_types.cpu().numpy()
    x_lane_states = x_lane_states.cpu().numpy()
    x_lane_conn = x_lane_conn.cpu().numpy()
    agent_batch = agent_batch.cpu().numpy()
    lane_batch = lane_batch.cpu().numpy()
    lane_conn_batch = lane_conn_batch.cpu().numpy()

    batch_of_scenarios = {}
    for i in range(batch_size):
        map_id_i = map_ids[i]  
        scene_i_agents = x_agent_states[agent_batch == i]
        scene_i_lanes = x_lane_states[lane_batch == i]
        scene_i_agent_types = x_agent_types[agent_batch == i]
        if cache_lane_types:
            scene_i_lane_types = x_lane_types[lane_batch == i]
        scene_i_lane_conns = x_lane_conn[lane_conn_batch == i]
        lg_type_i = lg_type[i]
        
        data = {
            'num_agents': len(scene_i_agents),
            'num_lanes': len(scene_i_lanes),
            'map_id': map_id_i,
            'lg_type': lg_type_i,
            'agent_states': scene_i_agents,
            'road_points': scene_i_lanes,
            'agent_types': scene_i_agent_types,
            'road_connection_types': scene_i_lane_conns
        }

        if cache_lane_types:
            data['lane_types'] = scene_i_lane_types

        scenario_id = f"{i}_{batch_idx}"
        filename = f"{scenario_id}.pkl"

        batch_of_scenarios[scenario_id] = data
        
        if cache_samples:
            with open(os.path.join(cache_dir, filename), 'wb') as f:
                pickle.dump(data, f)

    return batch_of_scenarios