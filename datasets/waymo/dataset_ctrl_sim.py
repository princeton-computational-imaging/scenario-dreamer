import os
import hydra
import glob
import torch
import pickle
import random
import copy
from tqdm import tqdm

from torch_geometric.data import Dataset
import torch.nn.functional as F
import numpy as np

from cfgs.config import CONFIG_PATH
from utils.lane_graph_helpers import resample_polyline, get_compact_lane_graph
from utils.data_helpers import add_batch_dim, extract_raw_waymo_data
from utils.torch_helpers import from_numpy
from utils.data_container import CtRLSimData
from utils.geometry import apply_se2_transform, normalize_agents, normalize_angle
from utils.collision_helpers import batched_collision_checker
from utils.k_disks_helpers import (
    transform_box_corners_from_vocab,
    get_local_state_transition,
    transform_box_corners_from_local_state,
    get_global_next_state
)

class CtRLSimDataset(Dataset):
    # agent_states: [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
    POS_X_IDX = 0
    POS_Y_IDX = 1
    VEL_X_IDX = 2
    VEL_Y_IDX = 3
    HEAD_IDX = 4
    LEN_IDX = 5
    WID_IDX = 6
    EXIST_IDX = -1
    # In Waymo, AV is last agent
    AV_IDX = -1

    def __init__(self, cfg, split_name='train'):
        super(CtRLSimDataset, self).__init__()

        self.cfg = cfg
        self.data_root = self.cfg.dataset_path
        self.split_name = split_name
        self.preprocess = self.cfg.preprocess
        self.delta_t = 1 / self.cfg.simulation_hz

        self.preprocessed_dir = os.path.join(self.cfg.preprocess_dir, f"{self.split_name}")
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        if not self.preprocess:
            self.files = glob.glob(
                os.path.join(
                    self.data_root, f"{self.split_name}"
                ) + "/*.pkl"
            )
        else:
            self.files = glob.glob(
                os.path.join(
                    self.preprocessed_dir
                ) + "/*.pkl"
            )
        
        self.files = sorted(self.files)
        self.dset_len = len(self.files)
        # Shuffle files for collecting state transitions to build K-disks vocabulary
        if self.cfg.collect_state_transitions:
            random.shuffle(self.files)
        else:
            with open(self.cfg.k_disks_vocab_path, 'rb') as f:
                self.V = np.array(pickle.load(f)['V'])
                print(f"Loaded K-disks vocabulary from {self.cfg.k_disks_vocab_path}, V shape: {self.V.shape}")


    def get_upsampled_and_sd_lanes(self, compact_lane_graph):
        """ Upsample lane polylines to a high resolution for precise offroad checks,
        then downsample to fixed number of points for model input compatible with Scenario Dreamer.
        """
        upsampled_lanes = []
        sd_lanes = []
        for lane_id in compact_lane_graph['lanes']:
            lane = compact_lane_graph['lanes'][lane_id]
            upsampled_lane = resample_polyline(lane, num_points=self.cfg.upsample_lane_num_points)
            sd_lane = resample_polyline(upsampled_lane, num_points=self.cfg.num_points_per_lane)
            upsampled_lanes.append(upsampled_lane)
            sd_lanes.append(sd_lane)
        upsampled_lanes = np.array(upsampled_lanes)
        sd_lanes = np.array(sd_lanes)
        return upsampled_lanes, sd_lanes
    

    def remove_offroad_agents(self, agent_states, agent_types, lanes):
        """ Remove agents that are offroad based on distance to nearest lane.
        
        This function differs from dataset_autoencoder_waymo.remove_offroad_agents
        in that the ego is the last agent, not the first.
        """
        
        # keep the ego vehicle always
        non_ego_agent_states = agent_states[:-1]
        non_ego_agent_types = agent_types[:-1]

        agent_road_dist = np.linalg.norm(non_ego_agent_states[:, :1, :2] - lanes.reshape(-1, 2)[np.newaxis, :, :], axis=-1).min(1)
        offroad_mask = agent_road_dist > self.cfg.offroad_threshold
        vehicle_mask = non_ego_agent_types[:, 1].astype(bool)
        offroad_vehicle_mask = offroad_mask * vehicle_mask

        onroad_agents = np.where(~offroad_vehicle_mask)[0]
        new_agent_states = np.concatenate([non_ego_agent_states[onroad_agents], agent_states[-1:]], axis=0)
        new_agent_types = np.concatenate([non_ego_agent_types[onroad_agents], agent_types[-1:]], axis=0)

        return new_agent_states, new_agent_types

    
    def rollout_k_disks(self, agent_states):
        """ Compute states and discrete actions based on K-disks tokenization rollout."""
        
        num_agents = agent_states.shape[0]
        num_steps = agent_states.shape[1] - 1

        states = np.zeros_like(agent_states)
        # discrete actions
        actions = np.zeros((num_agents, num_steps))

        states[:, 0] = agent_states[:, 0]
        for t in range(num_steps):
            valid_timestep = np.logical_and(
                agent_states[:, t, self.EXIST_IDX],
                agent_states[:, t+1, self.EXIST_IDX]
            )
            states[:, t, self.EXIST_IDX] = valid_timestep.astype(int)

            corner_0_x = - 1 * states[:, t, self.LEN_IDX] / 2
            corner_0_y = - 1 * states[:, t, self.WID_IDX] / 2
            corner_1_x = - 1 * states[:, t, self.LEN_IDX] / 2
            corner_1_y = states[:, t, self.WID_IDX] / 2
            corner_2_x = states[:, t, self.LEN_IDX] / 2
            corner_2_y = states[:, t, self.WID_IDX] / 2
            corner_3_x = states[:, t, self.LEN_IDX] / 2
            corner_3_y = -1 * states[:, t, self.WID_IDX] / 2

            box_corners = np.array([
                [corner_0_x, corner_0_y],
                [corner_1_x, corner_1_y],
                [corner_2_x, corner_2_y],
                [corner_3_x, corner_3_y]
            ]).transpose(2, 0, 1)

            # box_corners: [A, 4, 2]
            # V: [384, 3]
            # returns: transformed_box_corners: [A, 384, 4, 2]
            box_corners_vocab = transform_box_corners_from_vocab(box_corners, self.V)

            current_state = states[:, t, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX]]
            gt_next_state = agent_states[:, t+1, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX]]
            
            # this function computes the relative motion that takes you from current_state 
            # to next_state expressed in the local coordinates system of current_state
            local_state_transitions = get_local_state_transition(
                current_state=current_state, 
                next_state=gt_next_state
            )
            
            # box_corners: [A, 4, 2]
            # local_state_transitions: [A, 3]
            # returns: transformed_box_corners: [A, 4, 2]
            box_corners_local_state = transform_box_corners_from_local_state(
                box_corners, 
                local_state_transitions
            )

            # compute error between each vocab box corner and local state box corner
            err = np.linalg.norm(
                box_corners_vocab - box_corners_local_state[:, None, :, :], 
                axis=-1).mean(2)

            # nucleus sampling for action selection
            if self.cfg.tokenize_with_nucleus_sampling:
                err_torch = torch.from_numpy(-err)
                action_probs = F.softmax(err_torch / self.cfg.tokenization_temperature, dim=1)
                sorted_probs, sorted_indices = torch.sort(action_probs, dim=-1, descending=True)
                
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                selected_actions = cum_probs < self.cfg.tokenization_nucleus
                # handle case where first token has prob > p
                selected_actions[:, 0] = True

                next_action_dis = torch.zeros_like(err_torch)
                next_action_dis.scatter_(1, sorted_indices, selected_actions * sorted_probs)
                next_action_dis = next_action_dis / next_action_dis.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                next_actions = torch.multinomial(next_action_dis, 1)[:, 0]
                next_actions = next_actions.numpy()
            else:
                next_actions = np.argmin(err, axis=1)
            
            next_actions[~valid_timestep] = 0
            actions[:, t] = next_actions
            
            # current_state: [A, 3]
            # state_transitions: [A, 3]
            # returns: next state in global frame: [A, 3]
            next_state_pos_heading = get_global_next_state(
                current_state, 
                self.V[next_actions])
            
            next_v = (next_state_pos_heading[:, :2] - current_state[:, :2]) / self.delta_t
            # existence mask will be overwritten in next iteration
            next_exists = np.zeros(num_agents).astype(int)
            next_state = np.array([
                next_state_pos_heading[:, 0], 
                next_state_pos_heading[:, 1],
                next_v[:, 0],
                next_v[:, 1],
                next_state_pos_heading[:, 2],
                states[:, t, self.LEN_IDX], # length should not change during trajectory
                states[:, t, self.WID_IDX], # width should not change during trajectory
                next_exists
            ]).transpose(1, 0)
            
            next_state[~valid_timestep] = 0
            states[:, t+1] = next_state 

        return states, actions
    

    def get_ego_collision_rewards(self, agent_states_all):
        """ Compute vehicle-ego collision rewards."""
        # [1, 90, 2]
        ego_state = agent_states_all[self.AV_IDX:, :, :]
        # [A-1, 90, 2]
        other_states = agent_states_all[:self.AV_IDX, :, :]

        veh_ego_collision_reward = batched_collision_checker(
            ego_state[:, :, [
                self.POS_X_IDX, 
                self.POS_Y_IDX, 
                self.HEAD_IDX, 
                self.LEN_IDX, 
                self.WID_IDX]], 
            other_states[:, :, [
                self.POS_X_IDX, 
                self.POS_Y_IDX, 
                self.HEAD_IDX, 
                self.LEN_IDX, 
                self.WID_IDX]]
        )

        # ego collision reward for ego is 0
        veh_ego_collision_reward = np.concatenate(
            [veh_ego_collision_reward, 
             np.zeros((1, veh_ego_collision_reward.shape[1])).astype(int)]
             , axis=0)
        # mask out rewards for non-existing agents
        veh_ego_collision_reward = veh_ego_collision_reward * agent_states_all[:, :, -1]
        
        return veh_ego_collision_reward
    

    def get_last_valid_positions(self, states):
        """ Get last valid positions of all agents in the scene."""
        num_agents = len(states)
        last_valid_positions = []
        for a in range(num_agents):
            last_exists_timestep = np.where(states[a, :, self.EXIST_IDX])[0][0] - 1
            last_valid_positions.append(states[a, last_exists_timestep, :2])

        return np.array(last_valid_positions)


    def get_agent_mask(self, agent_states, normalize_dict, fov=None):
        """ Get mask of agents within field of view."""
        if fov is None:
            fov = self.cfg.fov
        
        agent_states = normalize_agents(agent_states, normalize_dict)
        agent_states = agent_states[:, :, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX]]
        agent_mask = np.logical_and(
            np.logical_and(agent_states[:, :, self.POS_X_IDX] < fov/2,
            agent_states[:, :, self.POS_Y_IDX] < fov/2),
            np.logical_and(agent_states[:, :, self.POS_X_IDX] > -1 * fov/2,
            agent_states[:, :, self.POS_Y_IDX] > -1 * fov/2)
        )
        return agent_mask

    
    def compute_rtgs(self, rewards):
        """ Compute return-to-go (rtg) for all agents and timesteps."""
        rtgs = np.lib.stride_tricks.sliding_window_view(rewards, window_shape=self.cfg.value_fn_horizon, axis=1)
        rtgs = rtgs.sum(-1)

        rtgs_full = np.zeros_like(rewards)
        rtgs_full[:, :rtgs.shape[1]] = rtgs 
        rtgs = rtgs_full 
        
        # normalize rtgs to [0, 1]
        rtgs = ((np.clip(
                rtgs, 
                a_min=self.cfg.min_rtg_veh, 
                a_max=self.cfg.max_rtg_veh
            ) - self.cfg.min_rtg_veh)
            / (self.cfg.max_rtg_veh - self.cfg.min_rtg_veh
        ))

        return rtgs
    

    def select_closest_max_num_agents(
            self, 
            agent_states, 
            agent_types, 
            agent_mask, 
            actions, 
            rtgs, 
            rtg_mask, 
            moving_agent_mask, 
            origin_agent_idx, 
            timestep,
            active_agents=None):
        """ Select the closest max_num_agents to the origin agent at the given timestep."""
        origin_states = agent_states[origin_agent_idx, timestep, :2].reshape(1, -1)
        dist_to_origin = np.linalg.norm(origin_states - agent_states[:, timestep, :2], axis=-1)
        
        # a valid agent is one that exists in the FOV at some point during the context buffer
        if active_agents is None:
            exists_during_buffer = agent_mask.sum(-1) > 0
            valid_agents = np.where(exists_during_buffer)[0]
        # used during simulation
        else:
            valid_agents = np.concatenate(
                [np.array([0]).astype(int), # ego always valid
                 active_agents]
                 , axis=0
            )

        final_agent_states = np.zeros((self.cfg.max_num_agents, *agent_states[0].shape))
        final_agent_types = -np.ones((self.cfg.max_num_agents, *agent_types[0].shape))
        final_agent_mask = np.zeros((self.cfg.max_num_agents, *agent_mask[0].shape))
        final_actions = np.zeros((self.cfg.max_num_agents, *actions[0].shape))
        final_rtgs = np.zeros((self.cfg.max_num_agents, *rtgs[0].shape))
        final_rtg_mask = np.zeros((self.cfg.max_num_agents, *rtg_mask[0].shape))
        final_moving_agent_mask = np.zeros(self.cfg.max_num_agents)

        # TODO: this is awkward. Fix in later release.
        if active_agents is None:
            closest_ag_ids = np.argsort(dist_to_origin)[:self.cfg.max_num_agents]
            closest_ag_ids = np.intersect1d(closest_ag_ids, valid_agents)
        else:
            closest_ag_ids = np.argsort(dist_to_origin)
            closest_ag_ids = np.intersect1d(closest_ag_ids, valid_agents)[:self.cfg.max_num_agents]
        # shuffle ids so it is not ordered by distance
        if self.split_name == 'train':
            np.random.shuffle(closest_ag_ids)
        
        final_agent_states[:len(closest_ag_ids)] = agent_states[closest_ag_ids]
        final_agent_types[:len(closest_ag_ids)] = agent_types[closest_ag_ids]
        final_agent_mask[:len(closest_ag_ids)] = agent_mask[closest_ag_ids]
        final_actions[:len(closest_ag_ids)] = actions[closest_ag_ids]
        final_rtgs[:len(closest_ag_ids)] = rtgs[closest_ag_ids]
        final_rtg_mask[:len(closest_ag_ids)] = rtg_mask[closest_ag_ids]
        final_moving_agent_mask[:len(closest_ag_ids)] = moving_agent_mask[closest_ag_ids]
        # idx of origin agent in new state tensors
        new_origin_agent_idx = np.where(closest_ag_ids == origin_agent_idx)[0][0]
        return (final_agent_states, 
                final_agent_types, 
                final_agent_mask, 
                final_actions, 
                final_rtgs, 
                final_rtg_mask, 
                final_moving_agent_mask, 
                new_origin_agent_idx,
                closest_ag_ids)

    
    def get_normalized_lanes_in_fov(self, lanes, normalize_dict):
        """ Normalize lanes to new coordinate frame and 
        return lanes within field of view up to max_num_lanes."""
        
        yaw = normalize_dict['yaw']
        translation = normalize_dict['center']
        
        # add pi/2 so that ego points north (as consistent with scenario dreamer output)
        angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
        translation = translation[np.newaxis, np.newaxis, :]
        
        lanes = apply_se2_transform(coordinates=lanes,
                                    translation=translation,
                                    yaw=angle_of_rotation)
        
        lane_point_dists_x = np.abs(lanes[:, :, 0])
        lane_point_dists_y = np.abs(lanes[:, :, 1])
        lanes_within_fov = np.logical_and(
            lane_point_dists_x < self.cfg.lane_fov / 2,
            lane_point_dists_y < self.cfg.lane_fov / 2
        )
        valid_lane_mask = lanes_within_fov.sum(1) > 0
        lanes = lanes[valid_lane_mask]
        lane_mask = lanes_within_fov[valid_lane_mask]

        if len(lanes) > self.cfg.max_num_lanes:
            min_road_dist_to_orig = np.linalg.norm(lanes[:, :, :2], axis=-1).min(1)
            closest_roads_to_ego = np.argsort(min_road_dist_to_orig)[:self.cfg.max_num_lanes]
            final_lanes = lanes[closest_roads_to_ego]
            final_lane_mask = lane_mask[closest_roads_to_ego]
        else:
            final_lanes = np.zeros((self.cfg.max_num_lanes, *lanes.shape[1:]))
            final_lanes[:len(lanes)] = lanes
            final_lane_mask = np.zeros((self.cfg.max_num_lanes, *lane_mask.shape[1:]), dtype=bool)
            final_lane_mask[:len(lanes)] = lane_mask
        
        return final_lanes, final_lane_mask


    def discretize_rtgs(self, rtgs):
        """ Discretize rtgs into integer bins."""
        rtgs = np.round(
            rtgs * (self.cfg.rtg_discretization - 1)
        )
        return rtgs


    def collect_state_transitions(self, data):
        """ Collect state transitions for k-disks vocabulary generation."""
        agent_data = data['objects']
        agent_states_all, _ = extract_raw_waymo_data(agent_data)

        existence_mask = agent_states_all[:, :, self.EXIST_IDX] == 1
        valid_agent_timesteps = np.logical_and(
            existence_mask[:, :-1],
            existence_mask[:, 1:]
        )

        agent_states_all = agent_states_all[:, :, [self.POS_X_IDX,self.POS_Y_IDX,self.HEAD_IDX]]
        diff_pos_all = agent_states_all[:, 1:, :2] - agent_states_all[:, :-1, :2]
        diff_head_all = normalize_angle(agent_states_all[:, 1:, 2:] - agent_states_all[:, :-1, 2:])
        
        diff_pos_all_reshaped = diff_pos_all.reshape(-1, 2)
        # apply negative of rotation of src state
        rotations_reshaped = -1 * agent_states_all[:, :-1, 2].reshape(-1)
        cos_theta = np.cos(rotations_reshaped)
        sin_theta = np.sin(rotations_reshaped)
        rotation_matrices = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])  # Shape [2, 2, N]
        rotation_matrices = np.transpose(rotation_matrices, (2, 0, 1))  # Shape [N, 2, 2]
        
        rotated_diff_pos_all_reshaped = np.einsum('ijk,ik->ij', rotation_matrices, diff_pos_all_reshaped)
        diff_head_all_reshaped = diff_head_all.reshape(-1, 1)
        
        state_transitions = np.concatenate([rotated_diff_pos_all_reshaped, diff_head_all_reshaped], axis=-1)
        valid_agent_timesteps = valid_agent_timesteps.reshape(-1)
        state_transitions = state_transitions[valid_agent_timesteps]

        return state_transitions


    def get_data(self, data, idx):
        """ Load preprocessed data or preprocess raw data."""
        if self.preprocess:
            num_agents = data['num_agents']
            lanes = data['lanes']
            states = data['states']
            actions = data['actions']
            agent_types = data['agent_types']
            last_valid_positions = data['last_valid_positions']
            veh_ego_collision_reward = data['veh_ego_collision_rewards']
        
        else:
            agent_data = data['objects']
            states, agent_types = extract_raw_waymo_data(agent_data)

            # as we are defining actions, we need both 1st and 2nd timesteps to exist
            exists_first_timestep = np.logical_and(
                states[:, 0, self.EXIST_IDX] == 1,
                states[:, 1, self.EXIST_IDX] == 1
            )
            assert exists_first_timestep[self.AV_IDX] == 1

            states = states[exists_first_timestep]
            agent_types = agent_types[exists_first_timestep]

            # handle missing data by setting existence to 0 from first missing timestep onward
            num_agents = states.shape[0]
            for a in range(num_agents):
                missing_indices = np.where(states[a, :, self.EXIST_IDX] == 0)[0]
                if len(missing_indices) > 0:
                    first_missing = missing_indices[0]
                    states[a, first_missing:, self.EXIST_IDX] = 0
            
            compact_lane_graph = get_compact_lane_graph(copy.deepcopy(data))
            # upsampled lane points necessary for precise offroad checks
            # lanes at resolution compatible with Scenario Dreamer
            lanes_upsampled, lanes = self.get_upsampled_and_sd_lanes(compact_lane_graph)

            # remove vehicles that are offroad (>2.5m from lane centerline) 
            # at the initial timestep, but keep ego vehicle
            states, agent_types = self.remove_offroad_agents(
                states, 
                agent_types, 
                lanes_upsampled
            )

            # rollout states and actions with discretized action space
            states, actions = self.rollout_k_disks(copy.deepcopy(states))
            num_agents = len(states)

            veh_ego_collision_reward = self.get_ego_collision_rewards(states)
            last_valid_positions = self.get_last_valid_positions(states)

            # save preprocessed data for accelerated training
            to_pickle = dict()
            to_pickle['idx'] = idx
            to_pickle['num_agents'] = num_agents 
            to_pickle['lanes'] = lanes
            to_pickle['states'] = states
            to_pickle['actions'] = actions
            to_pickle['agent_types'] = agent_types
            to_pickle['veh_ego_collision_rewards'] = veh_ego_collision_reward
            to_pickle['last_valid_positions'] = last_valid_positions

            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
            with open(os.path.join(self.preprocessed_dir, f'{raw_file_name}.pkl'), 'wb') as f:
                pickle.dump(to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        # identify moving agents based on displacement from last valid position
        moving_agents = np.where(
            np.linalg.norm(
                states[:, 0, :2] - last_valid_positions[:, :2], axis=1
            ) >= self.cfg.moving_threshold)[0]
        # always normalize to ego
        origin_idx = num_agents - 1
        # timesteps where ego exists
        valid_timesteps = np.where(states[origin_idx, :, self.EXIST_IDX] == 1)[0]
        max_timestep = max(np.max(valid_timesteps) - (self.cfg.train_context_length - 1), 0)
        
        # timestep of start of chunk
        start_timestep = random.randint(0, max_timestep)
        # normalize chunk to random timestep
        if self.cfg.normalize_to_random_timestep:
            normalize_timestep = np.random.randint(
                start_timestep, 
                min(start_timestep+31, np.max(valid_timesteps))
            )
            relative_normalize_timestep = normalize_timestep - start_timestep
        # normalize to last timestep
        else:
            normalize_timestep = min(
                start_timestep+31, 
                np.max(valid_timesteps)
            )
            relative_normalize_timestep = normalize_timestep - start_timestep
        
        normalize_dict = {
            'center': states[origin_idx, normalize_timestep, :self.POS_Y_IDX+1].copy(),
            'yaw': states[origin_idx, normalize_timestep, self.HEAD_IDX].copy()
        }
        
        # relative timesteps for chunk
        timesteps = np.arange(self.cfg.train_context_length)
        agent_mask = self.get_agent_mask(
            copy.deepcopy(states[:, :, :self.HEAD_IDX+1]), 
            normalize_dict
        )
        rewards = (
            -1 * veh_ego_collision_reward * self.cfg.rew_multiplier
        ) * states[:, :, self.EXIST_IDX]
        rtgs = self.compute_rtgs(rewards)
        rtgs = self.discretize_rtgs(rtgs)
        rtg_mask = np.ones(rtgs.shape, dtype=bool)

        # grab data chunk
        timestep_buffer = np.repeat(
            timesteps[np.newaxis, :, np.newaxis], 
            self.cfg.max_num_agents, 
            0
        )
        state_buffer = states[:, start_timestep:start_timestep+self.cfg.train_context_length]
        agent_type_buffer = agent_types
        agent_mask_buffer = agent_mask[:, start_timestep:start_timestep+self.cfg.train_context_length]
        action_buffer = actions[:, start_timestep:start_timestep+self.cfg.train_context_length]
        rtg_buffer = rtgs[:, start_timestep:start_timestep+self.cfg.train_context_length]
        rtg_mask_buffer = rtg_mask[:, start_timestep:start_timestep+self.cfg.train_context_length]
        moving_agent_mask = np.isin(np.arange(num_agents), moving_agents)

        # select the closest <= max_num_agents agents and modify data buffers accordingly
        (state_buffer, 
         agent_type_buffer, 
         agent_mask_buffer, 
         action_buffer, 
         rtg_buffer, 
         rtg_mask_buffer, 
         moving_agent_mask, 
         new_origin_agent_idx,
         _) = self.select_closest_max_num_agents(
             state_buffer, 
             agent_type_buffer, 
             agent_mask_buffer, 
             action_buffer, 
             rtg_buffer, 
             rtg_mask_buffer, 
             moving_agent_mask, 
             origin_agent_idx=origin_idx, 
             timestep=relative_normalize_timestep)

        # returns valid lane points (normalized to ego) 
        # and lane mask that identifies which points are within fov
        lanes, lanes_mask = self.get_normalized_lanes_in_fov(
            lanes, 
            normalize_dict
        )
        state_buffer = normalize_agents(
            state_buffer, 
            normalize_dict
        )
        num_centerlines = len(lanes)

        if num_centerlines == 0:
            d = CtRLSimData({})
            no_roadgraph = True 
        
        # identify the ego agent, add as categorical feature to states
        is_ego = np.zeros(len(state_buffer), dtype=int)
        is_ego[new_origin_agent_idx] = 1
        is_ego = np.tile(
            is_ego[:, None, None], 
            (1, self.cfg.train_context_length, 1)
        )

        # EXIST_IDX still last index
        state_buffer = np.concatenate([state_buffer[:, :, :-1], is_ego, state_buffer[:, :, -1:]], axis=-1)

        # filter out agents / lane positions that are not in the FOV
        # existence dimension account for both state not existing in dataset and in FOV
        state_buffer[~agent_mask_buffer.astype(bool)] = 0
        action_buffer[~agent_mask_buffer.astype(bool)] = 0
        rtg_buffer[~agent_mask_buffer.astype(bool)] = 0
        rtg_mask_buffer[~agent_mask_buffer.astype(bool)] = 0
        lanes[~lanes_mask.astype(bool)] = 0 
        # add mask into lane features
        lanes = np.concatenate([lanes, lanes_mask[:, :, None]], axis=-1)

        d = dict()
        d['idx'] = idx
        d['agent'] = from_numpy({
            'agent_states': add_batch_dim(state_buffer),
            'agent_types': add_batch_dim(agent_type_buffer), 
            'actions': add_batch_dim(action_buffer),
            'rtgs': add_batch_dim(rtg_buffer[:, :, None]),
            'rtg_mask': add_batch_dim(rtg_mask_buffer[:, :, None]),
            'timesteps': add_batch_dim(timestep_buffer),
            'moving_agent_mask': add_batch_dim(moving_agent_mask)
        })
        d['map'] = from_numpy({
            'road_points': add_batch_dim(lanes),
        })
        d = CtRLSimData(d)
        no_roadgraph = False

        return d, no_roadgraph


    def get(self, idx):
        # search for file with at least 2 agents
        if not self.cfg.preprocess:
            with open(self.files[idx], 'rb') as file:
                data = pickle.load(file)
                if len(data['objects']) == 1:
                    return None
            
            # collect state transitions for k-disks tokenization
            if self.cfg.collect_state_transitions:
                d = self.collect_state_transitions(data)
            else:
                d = self.get_data(data, idx)
        
        else:
            proceed = False
            while not proceed:
                # certain files may not have been preprocessed if they had only 1 agent
                raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
                raw_path = os.path.join(self.preprocessed_dir, f'{raw_file_name}.pkl')
                if os.path.exists(raw_path):
                    with open(raw_path, 'rb') as f:
                        data = pickle.load(f)
                    proceed = True
                else:
                    idx += 1
                
                if proceed:
                    d, no_roadgraph = self.get_data(data, idx)
                    # only load sample if it has a map
                    if no_roadgraph:
                        proceed = False
                        idx += 1
        return d


    def len(self):
        return self.dset_len


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    cfg.dataset_root = cfg.scratch_root
    cfg.ctrl_sim.dataset.preprocess = True
    cfg.ctrl_sim.dataset.preprocess_dir = os.path.join(cfg.dataset_root, 'scenario_dreamer_ctrl_sim_preprocess')
    
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    dset = CtRLSimDataset(cfg.ctrl_sim.dataset, split_name='train')
    idxs = list(range(len(dset)))
    random.shuffle(idxs)

    for i, idx in tqdm(enumerate(idxs)):
        with open(dset.files[idx], 'rb') as file:
            data = pickle.load(file)

        dset.get_data(data, idx)

        if i == 5000:
            break

if __name__ == '__main__':
    main()
