import os
import sys
import glob
import hydra
import torch
import pickle
import random
import math
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
torch.set_printoptions(threshold=100000)
import numpy as np
np.set_printoptions(suppress=True, threshold=sys.maxsize)
from cfgs.config import CONFIG_PATH

from utils.data import *
from utils.geometry import *

class WaymoDatasetAutoEncoder(Dataset):
    def __init__(self, cfg, split_name='train', mode='train'):
        super(WaymoDatasetAutoEncoder, self).__init__()
        self.cfg = cfg
        self.cfg_dataset = cfg.dataset.waymo
        self.data_root = self.cfg_dataset.dataset_path
        self.split_name = split_name 
        self.mode = mode
        self.preprocess = self.cfg_dataset.preprocess
        self.preprocessed_dir = os.path.join(self.cfg_dataset.preprocess_dir, f"{self.split_name}")
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir, exist_ok=True)

        if not self.preprocess:
            self.files = sorted(glob.glob(os.path.join(self.data_root, f"{self.split_name}") + "/*.pkl"))
            
            # TODO: remove?
            if self.split_name == 'test':
                self.files_augmented = copy.deepcopy(self.files)
                random.shuffle(self.files)
                # add 10000 more random files to get a large enough test set for evaluation
                self.files_augmented.extend(self.files[:10000])
                self.files = self.files_augmented
        else:
            self.files = sorted(glob.glob(self.preprocessed_dir + "/*.pkl"))
            
        self.dset_len = len(self.files)

    
    def extract_rawdata(self, agents_data):
        # Get indices of non-parked cars and cars that exist for the entire episode
        agent_data = []
        agent_types = []

        for n in range(len(agents_data)):
            ag_position = agents_data[n]['position']
            x_values = [entry['x'] for entry in ag_position]
            y_values = [entry['y'] for entry in ag_position]
            ag_position = np.column_stack((x_values, y_values))
            ag_heading = np.radians(np.array(agents_data[n]['heading']).reshape((-1, 1)))
            # wrap between -pi and pi
            ag_heading = np.mod(ag_heading + np.pi, 2 * np.pi) - np.pi
            ag_velocity = agents_data[n]['velocity']
            x_values = [entry['x'] for entry in ag_velocity]
            y_values = [entry['y'] for entry in ag_velocity]
            ag_velocity = np.column_stack((x_values, y_values))
            ag_existence = np.array(agents_data[n]['valid']).reshape((-1, 1))
            ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
            ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
            agent_type = get_object_type_onehot(agents_data[n]['type'])
            ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
            agent_data.append(ag_state)
            agent_types.append(agent_type)
        
        # convert to numpy array
        agent_data = np.array(agent_data)
        agent_types = np.array(agent_types)
        
        return agent_data, agent_types


    def get_compact_lane_graph(self, data):
        lane_ids = data['lane_graph']['lanes'].keys()
        pre_pairs = data['lane_graph']['pre_pairs']
        suc_pairs = data['lane_graph']['suc_pairs']
        left_pairs = data['lane_graph']['left_pairs']
        right_pairs = data['lane_graph']['right_pairs']

        for lid in pre_pairs.keys():
            lid1s = pre_pairs[lid]
            for lid1 in lid1s:
                if lid1 not in lane_ids:
                    pre_pairs[lid].remove(lid1)

        for lid in suc_pairs.keys():
            lid1s = suc_pairs[lid]
            for lid1 in lid1s:
                if lid1 not in lane_ids:
                    suc_pairs[lid].remove(lid1)

        for lid in left_pairs.keys():
            lid1s = left_pairs[lid]
            for lid1 in lid1s:
                if lid1 not in lane_ids:
                    left_pairs[lid].remove(lid1)

        for lid in right_pairs.keys():
            lid1s = right_pairs[lid]
            for lid1 in lid1s:
                if lid1 not in lane_ids:
                    right_pairs[lid].remove(lid1)

        for lane_id in lane_ids:
            if lane_id not in pre_pairs:
                pre_pairs[lane_id] = []
            if lane_id not in suc_pairs:
                suc_pairs[lane_id] = []
            if lane_id not in left_pairs:
                left_pairs[lane_id] = []
            if lane_id not in right_pairs:
                right_pairs[lane_id] = []

        lane_groups = find_lane_groups(pre_pairs, suc_pairs)       
        
        compact_lanes = {}
        compact_pre_pairs = {}
        compact_suc_pairs = {}
        compact_left_pairs = {}
        compact_right_pairs = {}
        
        for lane_group_id in lane_groups:
            compact_lane = []
            compact_pre_pair = []
            compact_suc_pair = []
            compact_left_pair = []
            compact_right_pair = []
            for i, lane_id in enumerate(lane_groups[lane_group_id]):
                if i == 0:
                    compact_lane.append(data['lane_graph']['lanes'][lane_id])
                    
                    if len(pre_pairs[lane_id]) > 0:
                        for pre_lane_id in pre_pairs[lane_id]:
                            compact_pre_pair.append(find_lane_group_id(pre_lane_id, lane_groups))
                else:
                    compact_lane.append(data['lane_graph']['lanes'][lane_id][1:])

                if len(left_pairs[lane_id]) > 0:
                    for left_lane_id in left_pairs[lane_id]:
                        to_append = find_lane_group_id(left_lane_id, lane_groups)
                        if to_append not in compact_left_pair:
                            compact_left_pair.append(to_append)
                
                if len(right_pairs[lane_id]) > 0:
                    for right_lane_id in right_pairs[lane_id]:
                        to_append = find_lane_group_id(right_lane_id, lane_groups)
                        if to_append not in compact_right_pair:
                            compact_right_pair.append(to_append)

                if i == len(lane_groups[lane_group_id]) - 1:
                    if len(suc_pairs[lane_id]) > 0:
                        for suc_lane_id in suc_pairs[lane_id]:
                            compact_suc_pair.append(find_lane_group_id(suc_lane_id, lane_groups))

            compact_lane = np.concatenate(compact_lane, axis=0)
            compact_lanes[lane_group_id] = compact_lane
            compact_pre_pairs[lane_group_id] = compact_pre_pair 
            compact_suc_pairs[lane_group_id] = compact_suc_pair 
            compact_left_pairs[lane_group_id] = compact_left_pair
            compact_right_pairs[lane_group_id] = compact_right_pair
        
        # NOTE: compact lane graph is now the full lane graph!
        compact_lane_graph = {
            'lanes': compact_lanes,
            'pre_pairs': compact_pre_pairs,
            'suc_pairs': compact_suc_pairs,
            'left_pairs': compact_left_pairs,
            'right_pairs': compact_right_pairs
        }

        return compact_lane_graph


    def partition_compact_lane_graph(self, compact_lane_graph):
        max_lane_id = max(list(compact_lane_graph['lanes'].keys()))
        next_lane_id = max_lane_id + 1

        lane_ids = list(compact_lane_graph['lanes'].keys())
        for lane_id in lane_ids:
            lane = compact_lane_graph['lanes'][lane_id]
            
            # Get y-values of the lane and find where it crosses or is near y = 0
            y_values = lane[:, 1]  # Assuming lane is [x, y] points
            sign_diff = np.insert(np.diff(np.signbit(y_values)), 0, 0)
            zero_crossings = np.where(sign_diff)[0]  # Indices where lane crosses y = 0
            
            if len(zero_crossings) == 0:  # If no crossings, skip this lane
                continue
            
            # Add artificial partitions at y = 0 crossings
            new_lanes = {}
            start_index = 0
            for crossing in zero_crossings:
                end_index = crossing + 1  # Create a partition from start to crossing
                new_lanes[next_lane_id] = lane[start_index:end_index]
                start_index = crossing  # Update start index for the next partition
                next_lane_id += 1
            
            # Handle the remaining part of the lane after the last crossing
            if zero_crossings[-1] < len(y_values) - 1:
                new_lanes[next_lane_id] = lane[start_index:]
                next_lane_id += 1
            
            # Update the compact_lane_graph with new lanes
            num_new_lanes = len(new_lanes)
            if num_new_lanes == 1:
                continue
            
            for j, new_lane_id in enumerate(new_lanes.keys()):
                compact_lane_graph['lanes'][new_lane_id] = new_lanes[new_lane_id]
                if j == 0:
                    compact_lane_graph['pre_pairs'][new_lane_id] = compact_lane_graph['pre_pairs'][lane_id]
                    # leveraging bijection between suc/pre
                    # replace successors of other lanes with new lane
                    for other_lane_id in compact_lane_graph['pre_pairs'][lane_id]:
                        if other_lane_id is not None:
                            compact_lane_graph['suc_pairs'][other_lane_id].remove(lane_id)
                            compact_lane_graph['suc_pairs'][other_lane_id].append(new_lane_id)
                    compact_lane_graph['suc_pairs'][new_lane_id] = [new_lane_id + 1] # by way we defined new lane ids
                
                elif j == num_new_lanes - 1:
                    compact_lane_graph['suc_pairs'][new_lane_id] = compact_lane_graph['suc_pairs'][lane_id]
                    # leveraging bijection between suc/pre
                    # replace predecessors of other lanes with new lane
                    for other_lane_id in compact_lane_graph['suc_pairs'][lane_id]:
                        if other_lane_id is not None:
                            compact_lane_graph['pre_pairs'][other_lane_id].remove(lane_id)
                            compact_lane_graph['pre_pairs'][other_lane_id].append(new_lane_id)
                    compact_lane_graph['pre_pairs'][new_lane_id] = [new_lane_id - 1] # by way we define new lane ids
                
                else:
                    compact_lane_graph['pre_pairs'][new_lane_id] = [new_lane_id - 1]
                    compact_lane_graph['suc_pairs'][new_lane_id] = [new_lane_id + 1]

                compact_lane_graph['left_pairs'][new_lane_id] = compact_lane_graph['left_pairs'][lane_id]
                compact_lane_graph['right_pairs'][new_lane_id] = compact_lane_graph['right_pairs'][lane_id]

            for other_lane_id in compact_lane_graph['right_pairs']:
                if lane_id in compact_lane_graph['right_pairs'][other_lane_id]:
                    compact_lane_graph['right_pairs'][other_lane_id].remove(lane_id)
                    for new_lane_id in new_lanes.keys():
                        compact_lane_graph['right_pairs'][other_lane_id].append(new_lane_id)

            for other_lane_id in compact_lane_graph['left_pairs']:
                if lane_id in compact_lane_graph['left_pairs'][other_lane_id]:
                    compact_lane_graph['left_pairs'][other_lane_id].remove(lane_id)
                    for new_lane_id in new_lanes.keys():
                        compact_lane_graph['left_pairs'][other_lane_id].append(new_lane_id)

            # remove old (now partitioned) lane from lane graph
            del compact_lane_graph['lanes'][lane_id]
            del compact_lane_graph['pre_pairs'][lane_id]
            del compact_lane_graph['suc_pairs'][lane_id]
            del compact_lane_graph['left_pairs'][lane_id]
            del compact_lane_graph['right_pairs'][lane_id]

        return compact_lane_graph


    def normalize_compact_lane_graph(self, lane_graph, normalize_dict):
        lane_ids = lane_graph['lanes'].keys()
        center = normalize_dict['center']
        angle_of_rotation = (np.pi / 2) + np.sign(-normalize_dict['yaw']) * np.abs(normalize_dict['yaw'])
        center = center[np.newaxis, np.newaxis, :]

        # normalize lanes to ego
        for lane_id in lane_ids:
            lane = lane_graph['lanes'][lane_id]
            lane = apply_se2_transform(coordinates=lane[:, np.newaxis, :],
                                       translation=center,
                                       yaw=angle_of_rotation)[:, 0]
            # overwrite with normalized lane (centered + rotated on AV)
            lane_graph['lanes'][lane_id] = lane
        
        return lane_graph


    def get_lane_graph_within_fov(self, lane_graph):
        lane_ids = lane_graph['lanes'].keys()
        pre_pairs = lane_graph['pre_pairs']
        suc_pairs = lane_graph['suc_pairs']
        left_pairs = lane_graph['left_pairs']
        right_pairs = lane_graph['right_pairs']
        
        lane_ids_within_fov = []
        valid_pts = {}
        for lane_id in lane_ids:
            lane = lane_graph['lanes'][lane_id]
            points_in_fov_x = np.abs(lane[:, 0]) < (self.cfg_dataset.fov / 2)
            points_in_fov_y = np.abs(lane[:, 1]) < (self.cfg_dataset.fov / 2)
            points_in_fov = points_in_fov_x * points_in_fov_y
            
            if np.any(points_in_fov):
                lane_ids_within_fov.append(lane_id)
                valid_pts[lane_id] = points_in_fov

        lanes_within_fov = {}
        pre_pairs_within_fov = {}
        suc_pairs_within_fov = {}
        left_pairs_within_fov = {}
        right_pairs_within_fov = {}
        for lane_id in lane_ids_within_fov:
            if lane_id in lane_ids:
                lane = lane_graph['lanes'][lane_id][valid_pts[lane_id]]
                resampled_lane = resample_polyline(lane, num_points=self.cfg_dataset.upsample_lane_num_points)
                lanes_within_fov[lane_id] = resampled_lane
            
            if lane_id in pre_pairs:
                pre_pairs_within_fov[lane_id] = [l for l in pre_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                pre_pairs_within_fov[lane_id] = []
            
            if lane_id in suc_pairs:
                suc_pairs_within_fov[lane_id] = [l for l in suc_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                suc_pairs_within_fov[lane_id] = [] 

            if lane_id in left_pairs:
                left_pairs_within_fov[lane_id] = [l for l in left_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                left_pairs_within_fov[lane_id] = []
            
            if lane_id in right_pairs:
                right_pairs_within_fov[lane_id] = [l for l in right_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                right_pairs_within_fov[lane_id] = []
        
        lane_graph_within_fov = {
            'lanes': lanes_within_fov,
            'pre_pairs': pre_pairs_within_fov,
            'suc_pairs': suc_pairs_within_fov,
            'left_pairs': left_pairs_within_fov,
            'right_pairs': right_pairs_within_fov
        }
        
        return lane_graph_within_fov

    
    def get_road_points_adj(self, compact_lane_graph):
        resampled_lanes = []
        idx_to_id = {}
        id_to_idx = {}
        i = 0
        for lane_id in compact_lane_graph['lanes']:
            lane = compact_lane_graph['lanes'][lane_id]
            resampled_lane = resample_polyline(lane, num_points=self.cfg_dataset.num_points_per_lane)
            resampled_lanes.append(resampled_lane)
            idx_to_id[i] = lane_id
            id_to_idx[lane_id] = i
            
            i += 1
        
        resampled_lanes = np.array(resampled_lanes)
        num_lanes = min(len(resampled_lanes), self.cfg_dataset.max_num_lanes)
        dist_to_origin = np.linalg.norm(resampled_lanes, axis=-1).min(1)
        closest_lane_ids = np.argsort(dist_to_origin)[:num_lanes]
        resampled_lanes = resampled_lanes[closest_lane_ids]

        # mapping from old idx to new index after ordering by distance
        idx_to_new_idx = {}
        new_idx_to_idx = {}
        for i, j in enumerate(closest_lane_ids):
            idx_to_new_idx[j] = i 
            new_idx_to_idx[i] = j

        pre_road_adj = np.zeros((num_lanes, num_lanes))
        suc_road_adj = np.zeros((num_lanes, num_lanes))
        left_road_adj = np.zeros((num_lanes, num_lanes))
        right_road_adj = np.zeros((num_lanes, num_lanes))
        for new_idx_i in range(num_lanes):
            for id_j in compact_lane_graph['pre_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    pre_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1 

            for id_j in compact_lane_graph['suc_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    suc_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

            for id_j in compact_lane_graph['left_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    left_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

            for id_j in compact_lane_graph['right_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    right_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

        return resampled_lanes, pre_road_adj, suc_road_adj, left_road_adj, right_road_adj, num_lanes


    def get_agents_within_fov(self, agent_states, agent_types, normalize_dict, av_index):
        center = normalize_dict['center']
        angle_of_rotation = (np.pi / 2) + np.sign(-normalize_dict['yaw']) * np.abs(normalize_dict['yaw'])
        center = center[np.newaxis, np.newaxis, :]

        agent_states[:, :2] = apply_se2_transform(coordinates=agent_states[:, np.newaxis, :2],
                                    translation=center,
                                    yaw=angle_of_rotation)[:, 0]
        agent_states[:, 2:4] = apply_se2_transform(coordinates=agent_states[:, np.newaxis, 2:4],
                                    translation=np.zeros_like(center),
                                    yaw=angle_of_rotation)[:, 0]
        agent_states[:, 4] = rotate_and_normalize_angles(agent_states[:, 4], angle_of_rotation)

        agents_in_fov_x = np.abs(agent_states[:, 0]) < (self.cfg_dataset.fov / 2)
        agents_in_fov_y = np.abs(agent_states[:, 1]) < (self.cfg_dataset.fov / 2)
        agents_in_fov_mask = agents_in_fov_x * agents_in_fov_y
        valid_agents = np.where(agents_in_fov_mask > 0)[0]
        
        dist_to_origin = np.linalg.norm(agent_states[:, :2], axis=-1)
        # up to max_num_agents agents
        closest_ag_ids = np.argsort(dist_to_origin)[:self.cfg_dataset.max_num_agents]
        closest_ag_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_agents)]

        new_agent_states = agent_states[closest_ag_ids]
        dist_to_origin = np.linalg.norm(new_agent_states[:, :2], axis=-1)

        return agent_states[closest_ag_ids], agent_types[closest_ag_ids]

    
    def remove_offroad_agents(self, agent_states, agent_types, lane_dict):
        # keep the ego vehicle always
        non_ego_agent_states = agent_states[1:]
        non_ego_agent_types = agent_types[1:]
        
        road_pts = []
        for lane_id in lane_dict:
            road_pts.append(lane_dict[lane_id])
        road_pts = np.concatenate(road_pts, axis=0)

        agent_road_dist = np.linalg.norm(non_ego_agent_states[:, np.newaxis, :2] - road_pts[np.newaxis, :, :], axis=-1).min(1)
        offroad_mask = agent_road_dist > self.cfg_dataset.offroad_threshold
        vehicle_mask = non_ego_agent_types[:, 1].astype(bool)
        offroad_vehicle_mask = offroad_mask * vehicle_mask

        onroad_agents = np.where(~offroad_vehicle_mask)[0]

        new_agent_states = np.concatenate([agent_states[:1], non_ego_agent_states[onroad_agents]], axis=0)
        new_agent_types = np.concatenate([agent_types[:1], non_ego_agent_types[onroad_agents]], axis=0)

        return new_agent_states, new_agent_types


    # replace with velocity with speed and replace heading with cos(heading) and sin(heading)
    def modify_agent_states(self, agent_states):
        new_agent_states = np.zeros_like(agent_states)
        new_agent_states[:, :2] = agent_states[:, :2]
        new_agent_states[:, 5:] = agent_states[:, 5:]
        new_agent_states[:, 2] = np.sqrt(agent_states[:, 2] ** 2 + agent_states[:, 3] ** 2)
        new_agent_states[:, 3] = np.cos(agent_states[:, 4])
        new_agent_states[:, 4] = np.sin(agent_states[:, 4])

        return new_agent_states


    def normalize_scene(self, agent_states, road_points):
        # pos_x
        agent_states[:, 0] = 2 * ((agent_states[:, 0] - (-1 * self.cfg_dataset.fov/2))
                             / self.cfg_dataset.fov) - 1
        # pos_y
        agent_states[:, 1] = 2 * ((agent_states[:, 1] - (-1 * self.cfg_dataset.fov/2))
                             / self.cfg_dataset.fov) - 1
        # speed
        agent_states[:, 2] = 2 * ((agent_states[:, 2] - (self.cfg_dataset.min_speed))
                             / (self.cfg_dataset.max_speed - self.cfg_dataset.min_speed)) - 1
        # length
        agent_states[:, 5] = 2 * ((agent_states[:, 5] - (self.cfg_dataset.min_length))
                             / (self.cfg_dataset.max_length - self.cfg_dataset.min_length)) - 1
        # width
        agent_states[:, 6] = 2 * ((agent_states[:, 6] - (self.cfg_dataset.min_width))
                             / (self.cfg_dataset.max_width - self.cfg_dataset.min_width)) - 1
        
        # road pos_x
        road_points[:, :, 0] = 2 * ((road_points[:, :, 0] - (self.cfg_dataset.min_lane_x))
                             / (self.cfg_dataset.max_lane_x - self.cfg_dataset.min_lane_x)) - 1
        road_points[:, :, 1] = 2 * ((road_points[:, :, 1] - (self.cfg_dataset.min_lane_y))
                             / (self.cfg_dataset.max_lane_y - self.cfg_dataset.min_lane_y)) - 1

        return agent_states, road_points


    def randomize_indices(self, agent_states, agent_types, road_points, edge_index_lane_to_lane):
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
        
        old_index_positions = np.argsort(lane_perm)
        edge_index_lane_to_lane_new = old_index_positions[edge_index_lane_to_lane]

        return agent_states, agent_types, road_points, edge_index_lane_to_lane_new

    
    def get_partitioned_masks(self, agents, lanes, a2a_edge_index, l2l_edge_index, l2a_edge_index):
        a2a_edge_index = a2a_edge_index.numpy()
        l2l_edge_index = l2l_edge_index.numpy()
        l2a_edge_index = l2a_edge_index.numpy()
        
        num_agents = len(agents)
        num_lanes = len(lanes)

        agents_y = agents[:, 1]
        lanes_y = lanes[:, 9, 1]
        agents_after_origin = np.where(agents_y > 0)[0]
        lanes_after_origin = np.where(lanes_y > 0)[0]

        # sum only equals 1 if two agents on opposite sides of partition
        a2a_mask = np.isin(a2a_edge_index, agents_after_origin).sum(0) != 1
        l2l_mask = np.isin(l2l_edge_index, lanes_after_origin).sum(0) != 1

        lane_l2a_mask = np.isin(l2a_edge_index[0], lanes_after_origin)[None, :]
        agent_l2a_mask = np.isin(l2a_edge_index[1], agents_after_origin)[None, :]
        l2a_mask = np.concatenate([lane_l2a_mask, agent_l2a_mask], axis=0).sum(0) != 1   

        return torch.from_numpy(a2a_mask), torch.from_numpy(l2l_mask), torch.from_numpy(l2a_mask), lanes_y <= 0
    
    
    def get_data(self, data, idx):
        if self.preprocess:
            road_points = data['road_points']
            agent_states = data['agent_states']
            edge_index_lane_to_lane = data['edge_index_lane_to_lane']
            edge_index_lane_to_agent = data['edge_index_lane_to_agent']
            edge_index_agent_to_agent = data['edge_index_agent_to_agent']
            road_connection_types = data['road_connection_types']
            num_lanes = data['num_lanes']
            num_agents = data['num_agents']
            agent_types = data['agent_types']
            lg_type = data['lg_type']
            
        else:
            av_index = data['av_idx']
            agent_data = data['objects']
            agent_states_all, agent_types_all = self.extract_rawdata(agent_data)

            # statistics here
            normalize_statistics = {}
            
            compact_lane_graph = self.get_compact_lane_graph(copy.deepcopy(data))
            
            valid_timesteps = np.where(agent_states_all[av_index, :, -1] == 1)[0]
            rand_idx = random.randrange(len(valid_timesteps))
            scene_timestep = valid_timesteps[rand_idx]
            
            # av does not exist, continue
            if not agent_states_all[av_index, scene_timestep, -1]:
                d = {
                'normalize_statistics': None,
                'valid_scene': False
                }
                return d
            
            normalize_dict = {
                'center': agent_states_all[av_index, scene_timestep, :2].copy(),
                'yaw': agent_states_all[av_index, scene_timestep, 4].copy()
            }
            compact_lane_graph_scene = self.normalize_compact_lane_graph(copy.deepcopy(compact_lane_graph), normalize_dict)
            compact_lane_graph_scene = self.get_lane_graph_within_fov(compact_lane_graph_scene)
            if len(compact_lane_graph_scene['lanes']) == 0:
                d = {
                'normalize_statistics': None,
                'valid_scene': False
                }
                return d
            compact_lane_graph_inpainting = self.partition_compact_lane_graph(copy.deepcopy(compact_lane_graph_scene))

            exists_mask = copy.deepcopy(agent_states_all[:, scene_timestep, -1]).astype(bool)
            if self.cfg_dataset.generate_only_vehicles:
                agent_mask = copy.deepcopy(agent_types_all[:, 1]).astype(bool)
            else:
                agent_mask = copy.deepcopy(agent_types_all[:, 1] + agent_types_all[:, 2] + agent_types_all[:, 3]).astype(bool)
            
            exists_mask = exists_mask * agent_mask
            # only generating the initial states
            agent_states = copy.deepcopy(agent_states_all[exists_mask, scene_timestep])
            agent_types = copy.deepcopy(agent_types_all[exists_mask])
            av_index_new = np.sum(exists_mask[:av_index])
            
            agent_states, agent_types = self.get_agents_within_fov(agent_states, agent_types, normalize_dict, av_index_new)

            if self.cfg_dataset.remove_offroad_agents:
                # this only removes offroad vehicles
                agent_states, agent_types = self.remove_offroad_agents(agent_states, agent_types, compact_lane_graph_scene['lanes'])
            
            # replace heading with cosine and sine of heading, replace velocity with speed
            agent_states = self.modify_agent_states(agent_states)
            num_agents = len(agent_states)
            
            if num_agents == 0 or len(compact_lane_graph_scene['lanes']) == 0:
                d = {
                'normalize_statistics': None,
                'valid_scene': False
                }
                return d
            
            # process both the regular lane graph and partitioned lane graph for enabling inpainting
            lg_dict = {
                'regular': compact_lane_graph_scene,   
                'partitioned': compact_lane_graph_inpainting
            }
            for lg_type in lg_dict.keys():
                lg = lg_dict[lg_type]
                road_points, pre_road_adj, suc_road_adj, left_road_adj, right_road_adj, num_lanes = self.get_road_points_adj(lg)
                
                # get edge information
                edge_index_lane_to_lane = get_edge_index_complete_graph(num_lanes)
                edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents)
                # NOTE: not need to do edge_index_agent_to_lane, since we simply need to transpose the edge_index_lane_to_agent
                edge_index_lane_to_agent = get_edge_index_bipartite(num_lanes, num_agents)
                
                road_connection_types = []
                for i in range(edge_index_lane_to_lane.shape[1]):
                    pre_conn_indicator = pre_road_adj[edge_index_lane_to_lane[1, i], edge_index_lane_to_lane[0, i]]
                    suc_conn_indicator = suc_road_adj[edge_index_lane_to_lane[1, i], edge_index_lane_to_lane[0, i]]
                    left_conn_indicator = left_road_adj[edge_index_lane_to_lane[1, i], edge_index_lane_to_lane[0, i]]
                    right_conn_indicator = right_road_adj[edge_index_lane_to_lane[1, i], edge_index_lane_to_lane[0, i]]
                    if edge_index_lane_to_lane[1, i] == edge_index_lane_to_lane[0, i]:
                        road_connection_types.append(get_lane_connection_type_onehot('self'))
                    elif pre_conn_indicator:
                        road_connection_types.append(get_lane_connection_type_onehot('pred'))
                    elif suc_conn_indicator:
                        road_connection_types.append(get_lane_connection_type_onehot('succ'))
                    elif left_conn_indicator:
                        road_connection_types.append(get_lane_connection_type_onehot('left'))
                    elif right_conn_indicator:
                        road_connection_types.append(get_lane_connection_type_onehot('right'))
                    else:
                        road_connection_types.append(get_lane_connection_type_onehot('none'))
                road_connection_types = np.array(road_connection_types)

                ### FOR TESTING VISUALIZATION PURPOSES
                # colors = ['black', 'silver', 'lightcoral', 'firebrick', 'red', 'coral', 'sienna', 'darkorange', 'gold', 'darkkhaki', 'olive', 'yellow', 'yellowgreen', 'chartreuse', 'forestgreen', 'turquoise', 'lightcyan', 'teal', 'aqua', 'deepskyblue', 'royalblue', 'navy', 'mediumpurple', 'indigo', 'violet', 'darkviolet', 'magenta', 'deeppink', 'pink']
                # ct = 0
                # for i in range(len(road_points)):
                #     lane = road_points[i, :, :2]
                #     if len(lane) == 0:
                #         continue
                    
                #     plt.plot(lane[:, 0], lane[:, 1], color=colors[ct % len(colors)], linewidth=1.5)
                #     # plt.plot(lane[:, 0], lane[:, 1], color='black', linewidth=1.5)
                #     ct += 1
                
                #     label_idx = len(lane) // 2
                #     plt.annotate(i,
                #         (lane[label_idx, 0], lane[label_idx, 1]), zorder=5, fontsize=5)
                
                #     for j in range(edge_index_lane_to_lane.shape[1]):
                #         if road_connection_types[j, 1] == 1: # or road_connection_types[j, 2] == 1 or road_connection_types[j, 3] == 1 or road_connection_types[j, 4] == 1:
                #             src_idx = edge_index_lane_to_lane[0, j]
                #             dest_idx = edge_index_lane_to_lane[1, j]

                #             lane_src = road_points[src_idx, :, :2]
                #             lane_dest = road_points[dest_idx, :, :2]
                #             src_pos = lane_src[10, :2]
                #             dest_pos = lane_dest[10, :2]

                #             edge_color = 'black'
                #             if road_connection_types[j, 2] == 1:
                #                 edge_color = 'red'
                #             elif road_connection_types[j, 3] == 1:
                #                 edge_color = 'green'
                #             elif road_connection_types[j, 4] == 1:
                #                 edge_color = 'blue'
                #             plt.arrow(src_pos[0], src_pos[1], dest_pos[0] - src_pos[0], dest_pos[1] - src_pos[1], length_includes_head=True, head_width=1, head_length=1, zorder=10, color=edge_color)
                
                
                # plt.scatter(agent_states[:, 0], agent_states[:, 1], s=10, color='black')
                # x_max = 32 
                # x_min = -32
                # y_max = 32 
                # y_min = -32
                # alpha = 0.25
                # edgecolor = 'black'
                # for a in range(len(agent_states)):
                #     if a == 0:
                #         color = 'blue'
                #     elif agent_types[a, 1]:
                #         color = '#ffde8b'
                #     elif agent_types[a, 2]:
                #         color = 'purple'
                #     elif agent_types[a, 3]:
                #         color = 'green'
                #     else:
                #         color = 'pink'
                    
                #     # draw bounding boxes
                #     length = agent_states[a, 5]
                #     width = agent_states[a, 6]
                #     bbox_x_min = agent_states[a, 0] - width / 2
                #     bbox_y_min = agent_states[a, 1] - length / 2
                #     lw = (0.35) / ((x_max - x_min) / 140)
                #     rectangle = mpatches.FancyBboxPatch((bbox_x_min, bbox_y_min),
                #                                 width, length, ec=edgecolor, fc=color, linewidth=lw, alpha=alpha,
                #                                 boxstyle=mpatches.BoxStyle("Round", pad=0.3))
                    
                #     cos_theta = agent_states[a, 3]
                #     sin_theta = agent_states[a, 4]
                #     theta = np.arctan2(sin_theta, cos_theta)
                #     # theta = np.arccos(cos_theta)
                #     tr = transforms.Affine2D().rotate_deg_around(agent_states[a, 0], agent_states[a, 1], radians_to_degrees(theta) - 90) + plt.gca().transData

                #     # Apply the transformation to the rectangle
                #     rectangle.set_transform(tr)
                    
                #     plt.gca().set_aspect('equal', adjustable='box')
                #     # Add the patch to the Axes
                #     plt.gca().add_patch(rectangle)
                    
                #     heading_length = length / 2 + 1.5
                #     heading_angle_rad = theta
                #     vehicle_center = agent_states[a, :2]

                #     # Calculate end point of the heading line
                #     line_end_x = vehicle_center[0] + heading_length * math.cos(heading_angle_rad)
                #     line_end_y = vehicle_center[1] + heading_length * math.sin(heading_angle_rad)

                #     # Draw the heading line
                #     plt.plot([vehicle_center[0], line_end_x], [vehicle_center[1], line_end_y], color='black', alpha=0.25, linewidth=0.25 / ((x_max - x_min) / 140))
                
                # plt.savefig('scene_{}_{}.png'.format(idx, 0 if lg_type == 'regular' else 1), dpi=1000)
                # plt.clf()
                
                raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
                to_pickle = dict()
                to_pickle['idx'] = idx
                to_pickle['lg_type'] = 0 if lg_type == 'regular' else 1
                to_pickle['scene_timestep'] = scene_timestep
                to_pickle['num_agents'] = num_agents 
                to_pickle['num_lanes'] = num_lanes
                to_pickle['road_points'] = road_points
                to_pickle['agent_states'] = agent_states[:, :-1] # no need for existence dimension
                to_pickle['agent_types'] = agent_types[:, 1:4] # only vehicle, pedestrian, bicyclist
                to_pickle['edge_index_lane_to_lane'] = edge_index_lane_to_lane
                to_pickle['edge_index_agent_to_agent'] = edge_index_agent_to_agent
                to_pickle['edge_index_lane_to_agent'] = edge_index_lane_to_agent
                to_pickle['road_connection_types'] = road_connection_types
                # save preprocessed file
                with open(os.path.join(self.preprocessed_dir, f'{raw_file_name}_{to_pickle["lg_type"]}_{to_pickle["scene_timestep"]}.pkl'), 'wb') as f:
                    pickle.dump(to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)

                if lg_type == 'regular':
                    normalize_statistics['max_speed'] = agent_states[:, 2].max()
                    normalize_statistics['min_length'] = agent_states[:, 5].min()
                    normalize_statistics['max_length'] = agent_states[:, 5].max()
                    normalize_statistics['min_width'] = agent_states[:, 6].min()
                    normalize_statistics['max_width'] = agent_states[:, 6].max()
                    normalize_statistics['min_lane_x'] = road_points[:, 0].min()
                    normalize_statistics['min_lane_y'] = road_points[:, 1].min()
                    normalize_statistics['max_lane_x'] = road_points[:, 0].max()
                    normalize_statistics['max_lane_y'] = road_points[:, 1].max()
            
            d = {
                'normalize_statistics': normalize_statistics,
                'valid_scene': True
            }

            return d
        
        agent_states, road_points = self.normalize_scene(agent_states, road_points)

        # randomize order of indices except for ego (which is always index 0)
        if self.mode == 'train':
            agent_states, agent_types, road_points, edge_index_lane_to_lane = self.randomize_indices(agent_states, agent_types, road_points, edge_index_lane_to_lane)
            edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)
        
        if lg_type == 1:
            a2a_mask, l2l_mask, l2a_mask, lane_partition_mask = self.get_partitioned_masks(agent_states, road_points, edge_index_agent_to_agent, edge_index_lane_to_lane, edge_index_lane_to_agent)
            agents_y = agent_states[:, 1]
            lanes_y = road_points[:, 9, 1]
            num_agents_after_origin = len(np.where(agents_y > 0)[0])
            num_lanes_after_origin = len(np.where(lanes_y > 0)[0])
        
        else:
            a2a_mask = torch.ones(edge_index_agent_to_agent.shape[1]).bool()
            l2l_mask = torch.ones(edge_index_lane_to_lane.shape[1]).bool()
            l2a_mask = torch.ones(edge_index_lane_to_agent.shape[1]).bool()
            lane_partition_mask = np.zeros(num_lanes).astype(bool)
            num_agents_after_origin = 0 
            num_lanes_after_origin = 0

        assert a2a_mask.shape[0] == edge_index_agent_to_agent.shape[1]
        assert l2l_mask.shape[0] == edge_index_lane_to_lane.shape[1]
        assert l2a_mask.shape[0] == edge_index_lane_to_agent.shape[1]
        assert lane_partition_mask.shape[0] == num_lanes

        if self.cfg_dataset.remove_left_right_connections: 
            # remove left and right connections for evaluation
            road_connection_types = road_connection_types[:, [0,1,2,5]]

        ### FOR TESTING VISUALIZATION PURPOSES
        # colors = ['black', 'silver', 'lightcoral', 'firebrick', 'red', 'coral', 'sienna', 'darkorange', 'gold', 'darkkhaki', 'olive', 'yellow', 'yellowgreen', 'chartreuse', 'forestgreen', 'turquoise', 'lightcyan', 'teal', 'aqua', 'deepskyblue', 'royalblue', 'navy', 'mediumpurple', 'indigo', 'violet', 'darkviolet', 'magenta', 'deeppink', 'pink']
        # ct = 0
        # for i in range(len(road_points)):
        #     lane = road_points[i, :, :2]
        #     if len(lane) == 0:
        #         continue
            
        #     plt.plot(lane[:, 0], lane[:, 1], color=colors[ct % len(colors)], linewidth=1.5)
        #     ct += 1
        
        #     label_idx = len(lane) // 2
        #     plt.annotate(i,
        #         (lane[label_idx, 0], lane[label_idx, 1]), zorder=5, fontsize=5)

        # for j in range(edge_index_lane_to_lane.shape[1]):
        #     if road_connection_types[j, 1] == 1: # or road_connection_types[j, 2] == 1 or road_connection_types[j, 3] == 1 or road_connection_types[j, 4] == 1:
        #         src_idx = edge_index_lane_to_lane[0, j]
        #         dest_idx = edge_index_lane_to_lane[1, j]

        #         print(j, src_idx, dest_idx)
                
        #         lane_src = road_points[src_idx, :, :2]
        #         lane_dest = road_points[dest_idx, :, :2]
        #         print(lane_src, lane_dest)
        #         exit()
        #         src_pos = lane_src[10, :2]
        #         dest_pos = lane_dest[10, :2]

        #         edge_color = 'black'
        #         if road_connection_types[j, 2] == 1:
        #             edge_color = 'red'
        #         elif road_connection_types[j, 3] == 1:
        #             edge_color = 'green'
        #         elif road_connection_types[j, 4] == 1:
        #             edge_color = 'blue'
        #         plt.arrow(src_pos[0], src_pos[1], dest_pos[0] - src_pos[0], dest_pos[1] - src_pos[1], length_includes_head=True, head_width=1, head_length=1, zorder=10, color=edge_color)

        # plt.savefig('scene_{}.png'.format(idx), dpi=1000)
        # plt.clf()
        
        
        d = dict()
        # need to add batch dim as pytorch_geometric batches along first dimension of torch Tensors
        d = ScenarioDreamerData()
        d['idx'] = idx
        d['num_lanes'] = num_lanes 
        d['num_agents'] = num_agents
        d['lg_type'] = lg_type
        # we no longer need the existence dimension
        d['agent'].x = from_numpy(agent_states)
        d['agent'].type = from_numpy(agent_types)
        d['lane'].x = from_numpy(road_points)
        d['lane'].partition_mask = from_numpy(lane_partition_mask)
        d['num_agents_after_origin'] = num_agents_after_origin 
        d['num_lanes_after_origin'] = num_lanes_after_origin

        # Assuming edge_index tensors for different edge types
        d['lane', 'to', 'lane'].edge_index = edge_index_lane_to_lane
        d['lane', 'to', 'lane'].type = torch.from_numpy(road_connection_types)
        d['agent', 'to', 'agent'].edge_index = edge_index_agent_to_agent
        d['lane', 'to', 'agent'].edge_index = edge_index_lane_to_agent
        d['lane', 'to', 'lane'].encoder_mask = l2l_mask
        d['lane', 'to', 'agent'].encoder_mask = l2a_mask
        d['agent', 'to', 'agent'].encoder_mask = a2a_mask

        return d
    
    
    def get(self, idx: int):
        if not self.cfg_dataset.preprocess:
            with open(self.files[idx], 'rb') as file:
                data = pickle.load(file)
            d = self.get_data(data, idx)

        else:
            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
            raw_path = os.path.join(self.preprocessed_dir, f'{raw_file_name}.pkl')
            with open(raw_path, 'rb') as f:
                data = pickle.load(f)
            d = self.get_data(data, idx)
        
        return d

    
    def len(self):
        return self.dset_len

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    dset = WaymoDatasetAutoEncoder(cfg, split_name='test')
    print(len(dset))
    np.random.seed(10)
    random.seed(10)
    torch.manual_seed(10)

    for idx in tqdm(range(len(dset))):
        with open(dset.files[idx], 'rb') as file:
            data = pickle.load(file)
        d = dset.get_data(data, idx)



if __name__ == '__main__':
    main()