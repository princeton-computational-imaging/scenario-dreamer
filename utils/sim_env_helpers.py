import os
import pickle
import numpy as np
import networkx as nx
import random
import torch

from cfgs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN
from utils.metrics_helpers import get_networkx_lane_graph, get_networkx_lane_graph_without_traffic_lights, get_lane_length
from utils.geometry import normalize_angle
from utils.lane_graph_helpers import resample_polyline
from utils.collision_helpers import batched_collision_checker, is_colliding
from utils.pyg_helpers import get_edge_index_complete_graph
from utils.viz import plot_scene


def clean_up_scene(data, dataset, mode='initial_scene', endpoint_threshold=1, offroad_threshold=2.5):
    """ Clean up the generated scene by removing duplicate lanes and colliding/offroad agents."""
    lanes = data['road_points']
    road_connection_types = data['road_connection_types']
    if dataset == 'nuplan':
        lane_types = data['lane_types']
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_NUPLAN
    else:
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_WAYMO
    num_lanes = data['num_lanes']
    # we include traffic light segments when removing duplicate lanes
    G, _ = get_networkx_lane_graph(data)

    if mode == 'inpainting':
        lanes_before_partition_mask = data['lane_mask'].astype(bool)
        lane_ids_after_partition = np.arange(num_lanes)[~lanes_before_partition_mask]
        cond_lane_ids = data['lane_ids'].astype(int)
    
    # for each lane, find lanes with same set of predecessors and successors
    node_groups = {}
    for node in G.nodes:
        if mode == 'inpainting' and (node not in lane_ids_after_partition):
            continue
        
        predecessors = set(G.predecessors(node))
        successors = set(G.successors(node))
        
        key = (frozenset(predecessors), frozenset(successors))
        
        if key not in node_groups:
            node_groups[key] = []
        node_groups[key].append(node)

    # find duplicate lanes based on endpoint proximity and predecessor/successor sets
    lanes_to_remove = []
    for key, lane_ids in node_groups.items():
        lane_ids_to_process = lane_ids.copy()
        while len(lane_ids_to_process) > 1:
            lane_id = np.array(lane_ids_to_process[:1])
            other = np.array(lane_ids_to_process[1:])

            start_lane_id = lanes[lane_id, 0]
            start_other = lanes[other, 0]
            
            end_lane_id = lanes[lane_id, -1]
            end_other = lanes[other, -1]

            start_close = np.linalg.norm(start_lane_id - start_other, axis=-1) < endpoint_threshold
            end_close = np.linalg.norm(end_lane_id - end_other, axis=-1) < endpoint_threshold
            lanes_close = start_close & end_close
            
            # ensure duplicated lanes are either both regular lanes or both traffic light segments
            # A traffic light segment over a lane is not considered a duplicate of the lane itself
            if dataset == 'nuplan':
                other_lane_ids = other[lanes_close]

                if other_lane_ids.size == 0:
                    both_lanes_or_both_traffic_light_segments = False
                elif np.argmax(lane_types[lane_id.item()]) == 0:
                    both_lanes_or_both_traffic_light_segments = np.any(np.argmax(lane_types[other_lane_ids], axis=-1) == 0)
                else:
                    both_lanes_or_both_traffic_light_segments = np.any(np.argmax(lane_types[other_lane_ids], axis=-1) != 0)
            else:
                both_lanes_or_both_traffic_light_segments = True

            if np.any(lanes_close) and both_lanes_or_both_traffic_light_segments:
                # only remove lanes with same set of pre/succ that are not near border
                # (two lanes may have the same predecessors (successors) and no successors (predecessors) simply because their endpoints are near the border)
                if not _near_border(lanes[lane_ids_to_process[0], -1]) and not _near_border(lanes[lane_ids_to_process[0], 0]):
                    lanes_to_remove.append(lane_ids_to_process[0])
            
            lane_ids_to_process.pop(0)

    # remove duplicate lanes and corresponding lane connectivity from data
    lanes_to_remove = np.array(lanes_to_remove)
    if len(lanes_to_remove) > 0:
        lane_graph_adj_pre = road_connection_types[:, LANE_CONNECTION_TYPES['pred']].reshape(num_lanes, num_lanes)
        lane_graph_adj_pre = np.delete(lane_graph_adj_pre, lanes_to_remove, axis=0)
        lane_graph_adj_pre = np.delete(lane_graph_adj_pre, lanes_to_remove, axis=1)

        lane_graph_adj_suc = road_connection_types[:, LANE_CONNECTION_TYPES['succ']].reshape(num_lanes, num_lanes)
        lane_graph_adj_suc = np.delete(lane_graph_adj_suc, lanes_to_remove, axis=0)
        lane_graph_adj_suc = np.delete(lane_graph_adj_suc, lanes_to_remove, axis=1)

        if dataset == 'waymo':
            lane_graph_adj_left = road_connection_types[:, LANE_CONNECTION_TYPES['left']].reshape(num_lanes, num_lanes)
            lane_graph_adj_left = np.delete(lane_graph_adj_left, lanes_to_remove, axis=0)
            lane_graph_adj_left = np.delete(lane_graph_adj_left, lanes_to_remove, axis=1)

            lane_graph_adj_right = road_connection_types[:, LANE_CONNECTION_TYPES['right']].reshape(num_lanes, num_lanes)
            lane_graph_adj_right = np.delete(lane_graph_adj_right, lanes_to_remove, axis=0)
            lane_graph_adj_right = np.delete(lane_graph_adj_right, lanes_to_remove, axis=1)

        lane_graph_pre = lane_graph_adj_pre.reshape(-1)
        lane_graph_suc = lane_graph_adj_suc.reshape(-1)
        lane_graph_self = torch.eye(len(lane_graph_adj_suc)).reshape(-1)
        if dataset == 'waymo':
            lane_graph_left = lane_graph_adj_left.reshape(-1)
            lane_graph_right = lane_graph_adj_right.reshape(-1)

        new_road_connection_types = np.zeros(len(lane_graph_suc)).astype(int)
        new_road_connection_types[lane_graph_pre == 1] = LANE_CONNECTION_TYPES['pred']
        new_road_connection_types[lane_graph_suc == 1] = LANE_CONNECTION_TYPES['succ']
        new_road_connection_types[lane_graph_self == 1] = LANE_CONNECTION_TYPES['self']
        if dataset == 'waymo':
            new_road_connection_types[lane_graph_left == 1] = LANE_CONNECTION_TYPES['left']
            new_road_connection_types[lane_graph_right == 1] = LANE_CONNECTION_TYPES['right']
        new_road_connection_types = np.eye(6 if dataset == 'waymo' else 4)[new_road_connection_types]

        valid_lane_ids = np.setdiff1d(np.arange(num_lanes), lanes_to_remove)
        data['road_points'] = lanes[valid_lane_ids]
        data['num_lanes'] = len(data['road_points'])
        data['road_connection_types'] = new_road_connection_types
        if dataset == 'nuplan':
            data['lane_types'] = lane_types[valid_lane_ids]
        if mode == 'inpainting':
            data['lane_mask'] = lanes_before_partition_mask[valid_lane_ids]
            data['lane_ids'] = cond_lane_ids[valid_lane_ids]

    # next, remove overlapping agents (all but one of the overlapping set) and offroad vehicles
    agents = data['agent_states']
    agent_types = data['agent_types']
    if mode == 'inpainting':
        agents_before_partition_mask = data['agent_mask'].astype(bool)
    num_agents = data['num_agents']
    # ego only existed superficially in inpainting mode
    agents_to_remove = [] if mode == 'initial_scene' else [0]
    agent_ids_to_process = np.arange(num_agents)[1:]
    while len(agent_ids_to_process) > 0:
        focal_agent_id = agent_ids_to_process[:1]
        
        # remove agents either colliding with other agents (later in sequence) or with ego
        if len(agent_ids_to_process) > 1:
            other_agent_ids = agent_ids_to_process[1:]
            if mode == 'initial_scene':
                np.append(other_agent_ids, [0])
            
            focal_agent = agents[focal_agent_id]
            other_agents = agents[other_agent_ids]
            collisions = batched_collision_checker(
                focal_agent[:, None, [0,1,4,5,6]], 
                other_agents[:, None, [0,1,4,5,6]])[:, 0].astype(bool)

            if np.any(collisions):
                agents_to_remove.append(focal_agent_id[0])

        # remove offroad (>offroad_threshold metres from lane) vehicles
        if np.argmax(agent_types[focal_agent_id[0]]) == 0:
            offroad = np.linalg.norm(agents[focal_agent_id, :2] - data['road_points'].reshape(-1, 2), axis=-1).min() > offroad_threshold

            if offroad:
                agents_to_remove.append(focal_agent_id[0])
        
        agent_ids_to_process = agent_ids_to_process[1:]

    if len(agents_to_remove) > 0:
        valid_agent_ids = np.setdiff1d(np.arange(num_agents), agents_to_remove)
        data['agent_states'] = agents[valid_agent_ids]
        data['agent_types'] = agent_types[valid_agent_ids]
        data['num_agents'] = len(data['agent_states'])
        if mode == 'inpainting':
            data['agent_mask'] = agents_before_partition_mask[valid_agent_ids]
    
    return data


def check_scene_validity(data, dataset):
    """ Check if the generated scene is valid based on several criteria."""
    
    if dataset == 'waymo':
        G, lanes = get_networkx_lane_graph(data)
    else:
        G, lanes = get_networkx_lane_graph_without_traffic_lights(data)

    # scenes are invalid if:
    # - they have lanes with no successors and not near border
    # - they have lanes with no predecessors and not near border
    passed_filter1 = True
    for lane_id, lane in enumerate(lanes):
        # no outgoing connections
        if G.out_degree(lane_id) == 0:
            if not _near_border(lane[-1]):
                passed_filter1 = False 
                break
        
        # no ingoing connections
        if G.in_degree(lane_id) == 0:
            if not _near_border(lane[0]):
                passed_filter1 = False 
                break 

    # scene is invalid if ego is more than 2.5 metre from route
    closest_lane_id = np.linalg.norm(lanes, axis=-1).min(1).argmin()
    closest_dist = np.linalg.norm(lanes[closest_lane_id], axis=-1).min()
    passed_filter2 = closest_dist <= 2.5
    
    # scene is invalid if there are discontinuities between connected lanes in the lane graph
    passed_filter3 = True
    for edge in G.edges():
        pre_endpoint = lanes[edge[0]][-1]
        suc_startpoint = lanes[edge[1]][0]

        if np.linalg.norm(pre_endpoint - suc_startpoint) > 1.5:
            passed_filter3 = False 

    return passed_filter1 and passed_filter2 and passed_filter3


def check_scene_validity_inpainting(data, dataset, heading_tolerance=np.pi/3):
    """ Check if the generated inpainted scene is valid based on several criteria."""

    if dataset == 'waymo':
        G, lanes = get_networkx_lane_graph(data)
        lane_before_partition_mask = data['lane_mask'].astype(bool)
    else:
        G, lanes = get_networkx_lane_graph_without_traffic_lights(data)
        lane_before_partition_mask = data['lane_mask'].astype(bool)[np.argmax(data['lane_types'], axis=1) == 0]

    # filter 1: scenes are invalid if:
    # - they have lane with endpoint near partition and no successors
    # - they have lane with startpoint near partition and no predecessors
    passed_filter1 = True
    lane_ids_with_endpoints_near_partition = []
    lane_ids_with_startpoints_near_partition = []
    heading_offset = np.pi / 2 if dataset == 'waymo' else 0
    for lane_id, lane in enumerate(lanes):
        first_heading, last_heading = estimate_heading(lane)
        # ignore lane if lane is near partition but does not have orientation that indicates it should connect with lane on other side of partition
        if _near_partition(lane[-1], dataset) and lane_before_partition_mask[lane_id] and np.abs(normalize_angle(last_heading - heading_offset)) < heading_tolerance:
            lane_ids_with_endpoints_near_partition.append(lane_id)
        elif _near_partition(lane[0], dataset) and lane_before_partition_mask[lane_id]  and np.abs(normalize_angle(first_heading - -heading_offset)) < heading_tolerance:
            lane_ids_with_startpoints_near_partition.append(lane_id)
    
    for lane_id in lane_ids_with_endpoints_near_partition:
        # no outgoing connections
        if G.out_degree(lane_id) == 0:
            passed_filter1 = False 
            break
    for lane_id in lane_ids_with_startpoints_near_partition:
        # no ingoing connections
        if G.in_degree(lane_id) == 0:
            passed_filter1 = False 
            break 
    
    # filter 2: scenes are invalid if:
    # - there are inpainted lanes with no successors that are not near border
    # - there are inpainted lanes with no predecessors that are not near border
    passed_filter2 = True
    inpainted_lanes = lanes[~lane_before_partition_mask]
    inpainted_lane_ids = np.arange(len(lanes))[~lane_before_partition_mask]
    for lane_id, lane in zip(inpainted_lane_ids, inpainted_lanes):
        # no outgoing connections
        if G.out_degree(lane_id) == 0:
            if not _near_border(lane[-1]):
                passed_filter2 = False 
                break
        # no ingoing connections
        if G.in_degree(lane_id) == 0:
            if not _near_border(lane[0]):
                passed_filter2 = False 
                break 
    
    # scene is invalid if there are discontinuities between connected lanes in the lane graph
    passed_filter3 = True
    for edge in G.edges():
        tolerance = 1.5
        # slightly more tolerance at partition
        if (edge[0] in lane_ids_with_endpoints_near_partition
            or edge[1] in lane_ids_with_startpoints_near_partition):
            tolerance = 2.5
        
        pre_endpoint = lanes[edge[0]][-1]
        suc_startpoint = lanes[edge[1]][0]

        if np.linalg.norm(pre_endpoint - suc_startpoint) > tolerance:
            passed_filter3 = False
    
    return passed_filter1, passed_filter2, passed_filter3


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


def sample_route(d, dataset, heading_tolerance=np.pi/3, num_points_in_route=1000):
    """ Sample a valid route for the ego vehicle in the scene."""
    
    if dataset == 'waymo':
        G, lanes = get_networkx_lane_graph(d)
    else:
        # route cannot consist of traffic-light segments
        G, lanes = get_networkx_lane_graph_without_traffic_lights(d)

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


def generate_simulation_environments(model, cfg, save_dir):
    """ Generate simulation environments using the trained Scenario Dreamer LDM model."""
    
    partial_samples_dir = os.path.join(save_dir, 'partial_sim_envs')
    complete_samples_dir = os.path.join(save_dir, 'complete_sim_envs')

    os.makedirs(partial_samples_dir, exist_ok=True)
    os.makedirs(complete_samples_dir, exist_ok=True)

    assert len(os.listdir(partial_samples_dir)) == 0, "Partial samples directory must be empty before generation."
    assert len(os.listdir(complete_samples_dir)) == 0, "Complete samples directory must be empty before generation."

     # Generate extra samples to account for potential failures
     # NuPlan exhibits more generation failures than Waymo, so we adjust accordingly
    max_num_samples = int(cfg.eval.num_samples * cfg.eval.sim_envs.overhead_factor) 

    print(f"Generating {cfg.eval.num_samples} simulation environments...")
    print(f"To account for degenerate samples, we will generate {max_num_samples} samples (overhead_factor={cfg.eval.sim_envs.overhead_factor}).")

    it = 0
    while len(os.listdir(complete_samples_dir)) < cfg.eval.num_samples:
        if it == 0:
            mode = 'initial_scene'
            num_iters = 1
            num_samples = max_num_samples
        else:
            mode = 'inpainting'
            num_iters = cfg.eval.sim_envs.num_inpainting_candidates
            num_samples = len(os.listdir(partial_samples_dir))
            if num_samples == 0:
                print("No partial samples available for inpainting. Ending generation.")
                break
        
        print(f"Iteration {it}: generating in mode {mode}...")
        candidate_next_samples = {}
        num_failed_check_1, num_failed_check_2, num_failed_check_3 = 0, 0, 0
        num_failed_found_route = 0
        num_failed_overlapping_tiles = 0
        for iter in range(num_iters):
            samples = model.generate(
                mode = mode,
                num_samples = num_samples,
                batch_size = cfg.eval.batch_size,
                cache_samples = False,
                visualize = False,
                conditioning_path = partial_samples_dir if mode == 'inpainting' else None,
                cache_dir = None,
                viz_dir = None,
                save_wandb = False,
                return_samples = True,
                nocturne_compatible_only = False if cfg.dataset_name == 'nuplan' else cfg.eval.sim_envs.nocturne_compatible_only
            )
            for sample in samples:
                if mode == 'initial_scene':
                    valid = check_scene_validity(samples[sample], cfg.dataset_name)
                    route, found_route = sample_route(samples[sample], cfg.dataset_name)
                    if found_route:
                        route_completed = get_lane_length(route) >= cfg.eval.sim_envs.route_length
                        # we track tile occupancy to ensure we don't have overlapping scenes in global frame
                        tile_corners = np.array([
                            [cfg.dataset.fov/2, cfg.dataset.fov/2],
                            [-cfg.dataset.fov/2, cfg.dataset.fov/2],
                            [-cfg.dataset.fov/2, -cfg.dataset.fov/2],
                            [cfg.dataset.fov/2, -cfg.dataset.fov/2]
                        ])
                        tile_occupancy = [tile_corners]
                        
                        if valid:
                            data = clean_up_scene(samples[sample], cfg.dataset_name, mode)
                            data['route'] = route
                            data['route_completed'] = route_completed
                            data['tile_occupancy'] = tile_occupancy

                            if cfg.eval.visualize:
                                plot_scene(
                                    data['agent_states'], 
                                    data['road_points'], 
                                    np.argmax(data['agent_types'], axis=1), 
                                    np.argmax(data['lane_types'], axis=1) if cfg.dataset_name == 'nuplan' else None,
                                    f"{it}_{sample}_{'PARTIAL' if not data['route_completed'] else 'COMPLETE'}.png", 
                                    os.path.join(save_dir, f"viz_sim_envs_{cfg.dataset_name}"), 
                                    return_fig=False,
                                    tile_occupancy=None,
                                    adaptive_limits=False,
                                    route=data['route'])


                            filename = f"{sample}.pkl"
                            if data['route_completed']:
                                write_dir = complete_samples_dir
                            else:
                                write_dir = partial_samples_dir
                            with open(os.path.join(write_dir, filename), 'wb') as f:
                                pickle.dump(data, f)
                
                else:
                    check_1, check_2, check_3 = check_scene_validity_inpainting(samples[sample], cfg.dataset_name)
                    valid = check_1 and check_2 and check_3
                    route, found_route = sample_route(samples[sample], cfg.dataset_name)

                    num_failed_check_1 += int(not check_1)
                    num_failed_check_2 += int(not check_2)
                    num_failed_check_3 += int(not check_3)
                    num_failed_found_route += int(not found_route)
                    
                    valid = valid and found_route
                    # determine is new tile overlaps with existing tiles
                    if valid:
                        with open(os.path.join(partial_samples_dir, f"{sample}.pkl"), 'rb') as f:
                            current_env = pickle.load(f)
                        existing_route = current_env['route']
                        tile_corners = np.array([
                                [cfg.dataset.fov/2, cfg.dataset.fov/2],
                                [-cfg.dataset.fov/2, cfg.dataset.fov/2],
                                [-cfg.dataset.fov/2, -cfg.dataset.fov/2],
                                [cfg.dataset.fov/2, -cfg.dataset.fov/2]
                            ])
                        _, last_heading = estimate_heading(existing_route)
                        transform_dict = {
                            'center': existing_route[-1],
                            'yaw': last_heading
                        }
                        transformed_tile_corners = _transform_corners(tile_corners, transform_dict, cfg.dataset_name)
                        overlapping_tiles = _check_overlapping_tiles(transformed_tile_corners, current_env['tile_occupancy'])
                        num_failed_overlapping_tiles += int(overlapping_tiles)
                        if not overlapping_tiles:
                            data = clean_up_scene(samples[sample], cfg.dataset_name, mode)
                            data['route'] = route
                            data['tile_occupancy'] = [transformed_tile_corners]

                            if sample not in candidate_next_samples:
                                candidate_next_samples[sample] = []
                            candidate_next_samples[sample].append(data)
        
        if it > 0:
            print(f"Number of failed validity check 1: {num_failed_check_1}")
            print(f"Number of failed validity check 2: {num_failed_check_2}")
            print(f"Number of failed validity check 3: {num_failed_check_3}")
            print(f"Number of failed to find route: {num_failed_found_route}")
            print(f"Number of failed overlapping tiles: {num_failed_overlapping_tiles}")
            
            new_envs = {}
            for sample in candidate_next_samples:
                with open(os.path.join(partial_samples_dir, f"{sample}.pkl"), 'rb') as f:
                    current_env = pickle.load(f)
                
                # select randomly from candidate extensions
                sampled_candidate = _sample_candidate(candidate_next_samples[sample], cfg.dataset_name)
                new_env = _extend_simulation_environment(current_env, sampled_candidate, cfg.eval.sim_envs.route_length, cfg.dataset_name)
                new_envs[sample] = new_env
            
            # first, clear out partial samples directory
            for filename in os.listdir(partial_samples_dir):
                file_path = os.path.join(partial_samples_dir, filename)
                os.remove(file_path)
            
            # then, write new partial samples
            for sample in new_envs:
                if len(os.listdir(complete_samples_dir)) >= cfg.eval.num_samples:
                    break
                
                filename = f"{sample}.pkl"
                if new_envs[sample]['route_completed']:
                    write_dir = complete_samples_dir
                else:
                    write_dir = partial_samples_dir
                with open(os.path.join(write_dir, filename), 'wb') as f:
                    pickle.dump(new_envs[sample], f)
        
            if cfg.eval.visualize:
                for sample in new_envs:
                    data = new_envs[sample]
                    plot_scene(
                        data['agent_states'], 
                        data['road_points'], 
                        np.argmax(data['agent_types'], axis=1), 
                        np.argmax(data['lane_types'], axis=1) if cfg.dataset_name == 'nuplan' else None,
                        f"{it}_{sample}_{'PARTIAL' if not data['route_completed'] else 'COMPLETE'}.png", 
                        os.path.join(save_dir, f"viz_sim_envs_{cfg.dataset_name}"), 
                        return_fig=False,
                        tile_occupancy=data['tile_occupancy'],
                        adaptive_limits=True,
                        route=data['route'])
        
        it += 1

    print(f"Generation completed. Generated {len(os.listdir(complete_samples_dir))} simulation environments.")


def _transform_scene(agents, lanes, route, transform_dict, dataset):
    """ Transform agents, lanes, and route according to the provided transformation dictionary."""
    yaw_offset = np.pi / 2 if dataset == 'waymo' else 0
    yaw = transform_dict['yaw']
    angle_of_rotation = normalize_angle(yaw - yaw_offset)
    translation = transform_dict['center']

    rotation_matrix = np.array([
        [np.cos(angle_of_rotation), -np.sin(angle_of_rotation)],
        [np.sin(angle_of_rotation), np.cos(angle_of_rotation)]
    ])

    # transform agents
    new_agents = np.zeros_like(agents)
    new_agents[:, :2] = np.dot(agents[:, :2], rotation_matrix.T) + translation
    cos_theta = agents[:, 3]
    sin_theta = agents[:, 4]
    theta = np.arctan2(sin_theta, cos_theta)
    new_theta = normalize_angle(theta + angle_of_rotation)
    new_agents[:, 2] = agents[:, 2] # speed remains unchanged
    new_agents[:, 3] = np.cos(new_theta)
    new_agents[:, 4] = np.sin(new_theta)
    new_agents[:, 5:] = agents[:, 5:] # other attributes remain unchanged

    # transform lanes
    new_lanes = np.dot(lanes.reshape(-1, 2), rotation_matrix.T) + translation
    new_lanes = new_lanes.reshape(lanes.shape)

    # transform route
    new_route = np.dot(route, rotation_matrix.T) + translation

    return new_agents, new_lanes, new_route


def _extend_simulation_environment(current_env, new_tile, target_route_length, dataset):
    """ Extend the current simulation environment with the new inpainted tile."""
    
    existing_route = current_env['route']
    _, last_heading = estimate_heading(existing_route)
    # transform tile corners be normalized at the endpoint of route
    transform_dict = {
        'center': existing_route[-1],
        'yaw': last_heading
    }
    new_agents, new_lanes, new_route = _transform_scene(
        new_tile['agent_states'][~new_tile['agent_mask'].astype(bool)],
        new_tile['road_points'][~new_tile['lane_mask'].astype(bool)],
        new_tile['route'],
        transform_dict,
        dataset,
    )
    
    # digraph for current env and new tile
    G_current_env = get_networkx_lane_graph(current_env)[0]
    
    current_env['agent_states'] = np.concatenate([
        current_env['agent_states'],
        new_agents
    ], axis=0)
    current_env['road_points'] = np.concatenate([
        current_env['road_points'],
        new_lanes
    ], axis=0)

    after_partition_lane_ids_new_tile = np.arange(len(new_tile['road_points']))[~new_tile['lane_mask'].astype(bool)]
    before_partition_lane_ids_new_tile = np.arange(len(new_tile['road_points']))[new_tile['lane_mask'].astype(bool)]

    # add new lanes to current env digraph
    num_new_lanes = len(new_lanes)
    for i in range(num_new_lanes):
        G_current_env.add_node(current_env['num_lanes'] + i)
    
    # find mapping between new tile digraph and augmented current env digraph
    new_tile_id_to_current_env_id = {}
    for lane_id in range(len(before_partition_lane_ids_new_tile)):
        new_tile_id_to_current_env_id[lane_id] = new_tile['lane_ids'][lane_id]
    for i, lane_id in enumerate(after_partition_lane_ids_new_tile):
        new_tile_id_to_current_env_id[lane_id] = current_env['num_lanes'] + i
    
    # mapping from edge to connection type
    road_connection_types_map_current_env = {}
    road_connection_types_current_env = current_env['road_connection_types']
    l2l_edge_index_current_env = get_edge_index_complete_graph(current_env['num_lanes']).transpose(1, 0)
    for i, edge in enumerate(l2l_edge_index_current_env):
        road_connection_types_map_current_env[(
            edge[0].item(), edge[1].item())] = (
                np.argmax(road_connection_types_current_env[i])
            )
    
    road_connection_types_map_new_tile = {}
    road_connection_types_new_tile = new_tile['road_connection_types']
    l2l_edge_index_new_tile = get_edge_index_complete_graph(len(new_tile['road_points'])).transpose(1, 0)
    for i, edge in enumerate(l2l_edge_index_new_tile):
        road_connection_types_map_new_tile[
            (new_tile_id_to_current_env_id[edge[0].item()], 
            new_tile_id_to_current_env_id[edge[1].item()])] = (
                np.argmax(road_connection_types_new_tile[i])
        )
    
    # build new road connection types for augmented env
    lane_ids_in_new_tile = list(new_tile_id_to_current_env_id.values())
    num_current_env_lanes = current_env['num_lanes']
    num_augmented_env_lanes = num_current_env_lanes + num_new_lanes
    l2l_edge_index_augmented_env = get_edge_index_complete_graph(num_augmented_env_lanes).transpose(1, 0)
    new_road_connection_types = np.zeros(len(l2l_edge_index_augmented_env))

    for i, edge in enumerate(l2l_edge_index_augmented_env):
        src = edge[0].item()
        dst = edge[1].item()
        # both lanes not in new tile, pull from current env
        if (src not in lane_ids_in_new_tile) and (dst not in lane_ids_in_new_tile):
            new_road_connection_types[i] = road_connection_types_map_current_env[(src, dst)]
        # both lanes in new tile, pull from new tile
        elif (src in lane_ids_in_new_tile) and (dst in lane_ids_in_new_tile):
            if src < num_current_env_lanes and dst < num_current_env_lanes:
                new_road_connection_types[i] = road_connection_types_map_current_env[(src, dst)]
            else:
                new_road_connection_types[i] = road_connection_types_map_new_tile[(src, dst)]
        # one lane in new tile, other not in new tile. Set to no-connection
        else:
            continue # already zero

    current_env['road_connection_types'] = np.eye(6 if dataset == 'waymo' else 4)[new_road_connection_types.astype(int)]
    current_env['agent_types'] = np.concatenate(
        [current_env['agent_types'],
        new_tile['agent_types'][~new_tile['agent_mask'].astype(bool)]],
        axis=0
    )
    if dataset == 'nuplan':
        current_env['lane_types'] = np.concatenate(
            [current_env['lane_types'],
            new_tile['lane_types'][~new_tile['lane_mask'].astype(bool)]],
            axis=0
        )
    
    current_env['route'] = np.concatenate(
        [existing_route,
        new_route],
        axis=0
    )

    current_env['num_agents'] = len(current_env['agent_states'])
    current_env['num_lanes'] = len(current_env['road_points'])
    current_env['route_completed'] = get_lane_length(current_env['route']) >= target_route_length
    current_env['tile_occupancy'].extend(new_tile['tile_occupancy'])

    return current_env


def _sample_candidate(candidates, dataset):
    """ Sample a candidate extension from multiple candidates.
    
    We prefer candidates that contain at least one vehicle.
    """
    candidates_with_vehicles = []
    num_lanes_in_candidates_with_vehicles = []
    for candidate in candidates:
        # vehicles are agent_type 0
        num_vehicles_in_candidate = (candidate['agent_types'] == 0).astype(int).sum()
        if num_vehicles_in_candidate > 0:
            candidates_with_vehicles.append(candidate)
            num_lanes_in_candidates_with_vehicles.append(len(candidate['road_points'][candidate['lane_mask'] == False]))
    
    if len(candidates_with_vehicles) == 0:
        sampled_candidate = random.sample(candidates, 1)[0]
    else:
        sampled_candidate = random.sample(candidates_with_vehicles, 1)[0]
        
    return sampled_candidate


def _near_border(pos, fov=64, threshold=1):
    """ Check if position is near border of FOV."""
    if np.abs(np.abs(pos[0]) - fov/2) < threshold or np.abs(np.abs(pos[1]) - fov/2) < threshold:
        return True 
    return False


def _near_partition(pos, dataset, threshold=2.5):
    """ Check if position is near partition (y=0)."""
    IDX = 1 if dataset == 'waymo' else 0
    return np.abs(pos[IDX]) < threshold


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


def _transform_corners(corners, transform_dict, dataset):
    """ Apply rotation and translation to corners."""
    
    yaw = transform_dict['yaw']
    angle_offset = np.pi / 2 if dataset == 'waymo' else 0
    angle_of_rotation = normalize_angle(yaw - angle_offset)
    translation = transform_dict['center']
    rotation_matrix = np.array([
        [np.cos(angle_of_rotation), -np.sin(angle_of_rotation)],
        [np.sin(angle_of_rotation), np.cos(angle_of_rotation)]
    ])
    rotated_corners = np.dot(corners, rotation_matrix.T)

    return rotated_corners + translation


def _check_overlapping_tiles(new_tile_corners, existing_tiles, ignore_last_n=3):
    """ Check if new tile overlaps with existing tiles."""

    overlapping = False
    for corners in existing_tiles[:-ignore_last_n]:
        if is_colliding(new_tile_corners, corners):
            overlapping = True
            break
    return overlapping