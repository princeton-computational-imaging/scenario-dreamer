import numpy as np
import torch
import networkx as nx
from utils.pyg_helpers import get_edge_index_complete_graph
from cfgs.config import NUPLAN_VEHICLE, UNIFIED_FORMAT_INDICES, NON_PARTITIONED
import torchaudio.functional as F
from scipy.spatial import distance
from utils.lane_graph_helpers import find_lane_groups, find_lane_group_id, resample_polyline, resample_lanes
from utils.sledge_helpers import calculate_progress, interpolate_path, coords_in_frame, find_consecutive_true_indices
from tqdm import tqdm

def compute_frechet_distance(X1, X2, apply_sqrt=True):
    """ Computes the (optionally square-root) Frechet distance between two sets of points X1 and X2."""
    mu_1 = np.mean(X1, axis=0)
    sigma_1 = np.cov(X1, rowvar=False)
    mu_2 = np.mean(X2, axis=0)
    sigma_2 = np.cov(X2, rowvar=False)
    
    if len(X1.shape) == 1:
        mu_1 = torch.tensor([mu_1])
        sigma_1 = torch.tensor(sigma_1).unsqueeze(0).unsqueeze(0)
        mu_2 = torch.tensor([mu_2])
        sigma_2 = torch.tensor(sigma_2).unsqueeze(0).unsqueeze(0)
    else:
        mu_1 = torch.tensor(mu_1)
        mu_2 = torch.tensor(mu_2)
        sigma_1 = torch.tensor(sigma_1)
        sigma_2 = torch.tensor(sigma_2)
    
    fd = F.frechet_distance(mu_1, sigma_1, mu_2, sigma_2).item()
    # Consistent with SLEDGE, which also computes the square root of the Frechet distance
    # This is done to make the Frechet distance more interpretable, as it is a distance metric
    if apply_sqrt:
        return np.sqrt(fd)
    else:
        return fd


def jsd(sim, gt, clip_min, clip_max, bin_size):
    """ Computes the Jensen-Shannon divergence (JSD) between generated (sim) and real (gt) distributions."""
    # Clip the simulated and ground truth values
    gt = np.clip(gt, clip_min, clip_max)
    sim = np.clip(sim, clip_min, clip_max)
    
    # Calculate bin edges based on the specified bin_size
    bin_edges = np.arange(clip_min, clip_max + bin_size, bin_size)
    
    # Compute the histograms and normalize to get probability distributions
    P = np.histogram(sim, bins=bin_edges)[0] / len(sim)
    Q = np.histogram(gt, bins=bin_edges)[0] / len(gt)
    
    # Compute Jensen-Shannon divergence and square it
    jsd_value = distance.jensenshannon(P, Q) ** 2  # Square to get the divergence
    return jsd_value


def compute_vehicle_circles(xy_position, heading, length, width):
    """ Computes the centroids and radii of circles around a vehicle based on its position, heading, length, and width."""
    num_circles = 5
    radius = width / 2
    relative_x_positions = np.linspace(-length / 2 + radius, length / 2 - radius, num_circles)
    
    # Compute the centroids of the circles
    # First, create the (x, y) relative offsets based on heading
    dx = np.cos(heading) * relative_x_positions
    dy = np.sin(heading) * relative_x_positions
    
    # Add these offsets to the vehicle's position to get the circle centroids
    centroids = np.column_stack((xy_position[0] + dx, xy_position[1] + dy))
    
    return centroids, np.array([radius]).repeat(num_circles)


def compute_collision_rate(samples):
    """ Computes the collision rate for the vehicles in the samples.
    Collision rate is computed by testing for overlapping circles around vehicles (with some threshold)."""
    print("Computing collision rate")
    num_vehicles_all = 0 
    num_vehicles_in_collision_all = 0
    for i in tqdm(range(len(samples))):
        data = samples[i]
        vehicles = data['vehicles']

        centroids_all = []
        radii_all = []
        for vehicle in vehicles:
            # vehicle: [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
            heading = np.arctan2(vehicle[UNIFIED_FORMAT_INDICES['sin_heading']], vehicle[UNIFIED_FORMAT_INDICES['cos_heading']])
            centroids, radii = compute_vehicle_circles(vehicle[:UNIFIED_FORMAT_INDICES['pos_y']+1], 
                                                       heading, 
                                                       vehicle[UNIFIED_FORMAT_INDICES['length']], 
                                                       vehicle[UNIFIED_FORMAT_INDICES['width']])
            centroids_all.append(centroids)
            radii_all.append(radii)
        centroids_all = np.array(centroids_all)
        radii_all = np.array(radii_all)

        num_vehicles_in_collision = 0
        for j in range(len(vehicles)):
            is_in_collision = False
            for k in range(len(vehicles)):
                if j == k:
                    continue
                
                thresh = (vehicles[j, 6] + vehicles[k, 6]) / np.sqrt(3.8)
                dist = np.linalg.norm(centroids_all[j, :, None] - centroids_all[k, None, :], axis=-1)
                bad = dist < thresh 
                if bad.sum() >= 1:
                    is_in_collision = True 
                    break
            
            if is_in_collision:
                num_vehicles_in_collision += 1
        
        num_vehicles_in_collision_all += num_vehicles_in_collision
        num_vehicles_all += len(vehicles)

    return num_vehicles_in_collision_all / num_vehicles_all


def get_compact_lane_graph(G, lanes, num_points_per_lane=20):
    """ Converts the lane graph to a compact representation by merging lanes that are connected and resampling them to a fixed number of points."""
    lanes_dict = {}
    pre_pairs = {}
    suc_pairs = {}
    for l, lane in enumerate(lanes):
        lanes_dict[l] = lane
        pre_pairs[l] = []
        suc_pairs[l] = []

    edges = G.edges()
    for edge in edges:
        pre_pairs[edge[1]].append(edge[0])
        suc_pairs[edge[0]].append(edge[1])

    lane_groups = find_lane_groups(pre_pairs, suc_pairs)     

    compact_lanes = {}
    compact_suc_pairs = {}
    for lane_group_id in lane_groups:
        compact_lane = []
        compact_suc_pair = []
        for i, lane_id in enumerate(lane_groups[lane_group_id]):
            if i == 0:
                compact_lane.append(lanes_dict[lane_id])
            else:
                compact_lane.append(lanes_dict[lane_id][1:])
            
            if i == len(lane_groups[lane_group_id]) - 1:
                if len(suc_pairs[lane_id]) > 0:
                    for suc_lane_id in suc_pairs[lane_id]:
                        compact_suc_pair.append(find_lane_group_id(suc_lane_id, lane_groups))

        compact_lane = np.concatenate(compact_lane, axis=0)
        compact_lanes[lane_group_id] = compact_lane
        compact_suc_pairs[lane_group_id] = compact_suc_pair
    
    idx_to_new_idx = {}
    for new_lane_id, lane_id in enumerate(compact_lanes.keys()):
        idx_to_new_idx[lane_id] = new_lane_id 
    
    compact_suc_pairs_reindexed = {}
    compact_lanes_all = []
    for lane_id in compact_lanes.keys():
        compact_suc_pairs_reindexed[idx_to_new_idx[lane_id]] = [idx_to_new_idx[idx] for idx in compact_suc_pairs[lane_id]]
        
        if len(compact_lanes[lane_id]) != num_points_per_lane:
            compact_lanes_all.append(resample_polyline(compact_lanes[lane_id], num_points=num_points_per_lane)[None, :, :])
        else:
            compact_lanes_all.append(compact_lanes[lane_id][None, :, :])

    compact_lanes_all = np.concatenate(compact_lanes_all, axis=0)

    num_lanes = len(compact_lanes_all)
    A = np.zeros((num_lanes, num_lanes))
    for lid in compact_suc_pairs_reindexed.keys():
        for suc_lid in compact_suc_pairs_reindexed[lid]:
            A[lid, suc_lid] = 1
    compact_G = nx.DiGraph(incoming_graph_data=A)

    return compact_G, compact_lanes_all


def _get_sledge_lane_graph_nuplan(data):
    """ Processes a nuPlan lane graph following the SLEDGE preprocessing scheme 
    to extract the lane connectivity and centerlines for metrics computation."""
    frame = (64, 64)
    pixel_size = 0.25 
    lines = data['lines']
    A = data['G']['states']
    G = nx.DiGraph(incoming_graph_data=A)

    lines_in_frame = []
    indices_to_remove = []
    for i, (line_states, line_mask) in enumerate(zip(lines['states'], lines['mask'])):
        line_in_mask = line_states[line_mask]  # (n, 3)
        if len(line_in_mask) < 2:
            indices_to_remove.append(i)
            continue

        path_progress = calculate_progress(line_in_mask)
        path_length = path_progress[-1]
        
        states_se2_array = line_in_mask
        states_se2_array[:, 2] = np.unwrap(states_se2_array[:, 2], axis=0)
        
        distances = np.arange(
            0,
            path_length + pixel_size,
            pixel_size,
        )
        line = interpolate_path(distances, path_length, path_progress, states_se2_array, as_array=True)

        frame_mask = coords_in_frame(line[..., :2], frame)
        indices_segments = find_consecutive_true_indices(frame_mask)
        line_segments = []
        for indices_segment in indices_segments:
            line_segment = line[indices_segment]
            if len(line_segment) < 3:
                continue
            line_segments.append(line_segment)
        
        if len(line_segments) > 0:
            lines_in_frame.append(line_segments)
        else:
            indices_to_remove.append(i)

    # relabel lane graph
    mapping = {}
    new_count = 0
    for idx in range(len(G)):
        if idx in indices_to_remove:
            continue 
        mapping[idx] = new_count
        new_count += 1

    # remove lanes from lane graph that are not in frame
    for idx in indices_to_remove:
        G.remove_node(idx)
    G = nx.relabel_nodes(G, mapping)

    inv_mapping = {}
    for old_idx, new_idx in mapping.items():
        inv_mapping[new_idx] = old_idx

    # remove edges where they connect outside the FOV
    edges_to_remove = []
    for edge in G.edges():
        line_states_src = data['lines']['states'][inv_mapping[edge[0]]]
        line_states_dst = data['lines']['states'][inv_mapping[edge[1]]]
        line_mask_src = data['lines']['mask'][inv_mapping[edge[0]]]
        line_mask_dst = data['lines']['mask'][inv_mapping[edge[1]]]
        src = line_states_src[line_mask_src][-1, :2]
        dst = line_states_dst[line_mask_dst][0, :2]
        
        if np.abs(src[0]) > 32 or np.abs(src[1]) > 32 or np.abs(dst[0]) > 32 or np.abs(dst[1]) > 32:
            edges_to_remove.append(edge)
    
    for edge in edges_to_remove:
        G.remove_edge(edge[0], edge[1])
    
    # add nodes for split lines
    final_lines_dict = {}
    num_lines_before_splitting = len(lines_in_frame)
    new_lines_count = 0
    edges_to_add = []
    for i, line_segments in enumerate(lines_in_frame):
        final_lines_dict[i] = line_segments[0]

        if len(line_segments) > 1:
            for j in range(len(line_segments[1:])):
                final_lines_dict[num_lines_before_splitting + new_lines_count] = line_segments[1 + j]
                G.add_node(num_lines_before_splitting + new_lines_count)
                if j == len(line_segments[1:]) - 1:
                    suc_nodes = list(G.successors(i))
                    for suc_node in suc_nodes:
                        G.remove_edge(i, suc_node)
                        edges_to_add.append((num_lines_before_splitting + new_lines_count, suc_node))
                
                new_lines_count += 1
    
    for edge in edges_to_add:
        G.add_edge(edge[0], edge[1])

    final_lines_list = []
    for i in range(len(G)):
        final_lines_list.append(final_lines_dict[i])

    lines_in_frame = final_lines_list

    vector_states = np.zeros((len(lines_in_frame), 20, 2), dtype=np.float32)
    for line_idx, line in enumerate(lines_in_frame):
        path_progress = calculate_progress(line)
        path_length = path_progress[-1]
        states_se2_array = line
        states_se2_array[:, 2] = np.unwrap(states_se2_array[:, 2], axis=0)
        distances = np.linspace(0, path_length, num=20, endpoint=True)
        vector_states[line_idx] = interpolate_path(distances, path_length, path_progress, states_se2_array, as_array=True)[..., :2]

    return G, vector_states


def get_lane_graph_nuplan(data):
    """ Processes a scenario dreamer generated nuPlan lane graph to extract the lane connectivity and centerlines
    for metrics computation."""
    num_lanes = data['num_lanes']
    l2l_edge_index = get_edge_index_complete_graph(num_lanes)
    lane_conn = data['road_connection_types']
    
    SUCC_IDX = 1
    is_succ = lane_conn[:, SUCC_IDX] == 1
    edges = l2l_edge_index[:, is_succ].transpose(1, 0)

    lane_types = np.argmax(data['lane_types'], axis=1)
    is_centerline = lane_types == 0
    traffic_lights = np.where(lane_types != 0)[0]

    # remove edges where incident lane is a traffic light
    edges_filtered = []
    for edge in edges:
        if edge[0].item() in traffic_lights or edge[1].item() in traffic_lights:
            continue
        edges_filtered.append(edge[None, :])
    if len(edges_filtered) > 0:
        edges = np.concatenate(edges_filtered, axis=0)
    else:
        edges = []

    lanes = data['road_points']
    centerlines = lanes[is_centerline]

    idx_to_new_idx = {}
    centerline_count = 0
    for i in range(num_lanes):
        if lane_types[i] == 0:
            idx_to_new_idx[i] = centerline_count 
            centerline_count += 1

    num_centerlines = len(centerlines)
    A = np.zeros((num_centerlines, num_centerlines))
    
    for edge in edges:
        A[idx_to_new_idx[edge[0].item()], idx_to_new_idx[edge[1].item()]] = 1

    G = nx.DiGraph(incoming_graph_data=A)

    return G, centerlines


def get_lane_graph_waymo(data):
    """ Processes a scenario dreamer or GT Waymo lane graph to 
    extract the lane connectivity and centerlines for metrics computation."""
    num_lanes = data['num_lanes']
    l2l_edge_index = get_edge_index_complete_graph(num_lanes)
    lane_conn = data['road_connection_types']
    
    # this seems counterintuitive as lane_conn indices are 
    # {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}
    # however, if [i,j] is of type pred (index 1), then that means i is 
    # a predecessor of j, i.e. j is a successor of i
    # which means the edge i --> j should be in the lane graph
    SUCC_IDX = 1
    is_succ = lane_conn[:, SUCC_IDX] == 1
    edges_succ = l2l_edge_index[:, is_succ].transpose(1, 0)

    centerlines = data['road_points']

    # for pre/succ connections
    num_centerlines = len(centerlines)
    A_succ = np.zeros((num_centerlines, num_centerlines))
    for edge in edges_succ:
        A_succ[edge[0].item(), edge[1].item()] = 1
    G_succ = nx.DiGraph(incoming_graph_data=A_succ)

    return G_succ, centerlines


def convert_data_to_unified_format(data, dataset_name):
    """ Converts the data from the dataset to a unified format for metrics computation."""
    
    # we are evaluating initial scenes, not inpainted scenes
    assert data['lg_type'] == NON_PARTITIONED
    
    if dataset_name == 'waymo' or dataset_name == 'waymo_gt':
        G_succ, centerlines = get_lane_graph_waymo(data)
    elif dataset_name == 'nuplan':
        G_succ, centerlines = get_lane_graph_nuplan(data)
    elif dataset_name == 'nuplan_gt':
        # We follow sledge preprocessing scheme for nuplan GT lane graphs (for fairest comparison with SLEDGE)
        G_succ, centerlines = _get_sledge_lane_graph_nuplan(data)

    agents = data['agent_states']
    agent_types = np.argmax(data['agent_types'], axis=1)
    
    # 0 corresponds to vehicle in waymo and nuplan
    # Although vehicle is index 1 in data_helpers.get_object_type_onehot_waymo
    # We index agent_types at [1:4] in line 1026 of datasets.waymo.dataset_autoencoder_waymo
    # Thus, 0 corresponds to vehicle, 1 to pedestrian, and 2 to bicycle
    vehicles = agents[agent_types == NUPLAN_VEHICLE]

    compact_G, compact_centerlines = get_compact_lane_graph(G_succ, centerlines)

    unified_data = {
        'G': compact_G, # networkX graph of lane connectivity
        'lanes': compact_centerlines, # [num_lanes, 20, 2]
        'vehicles': vehicles # [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
    }

    return unified_data


def get_lane_length(positions):
    """ Computes the length of a lane given its positions."""
    diffs = np.diff(positions, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    total_length = np.sum(distances)
    return total_length


def compute_route_length(samples):
    """ Computes the route length for each sample in the dataset."""
    print("Computing route lengths")
    num_samples_to_calculate_route_length = len(samples)
    path_lengths_all = []
    for i in tqdm(range(num_samples_to_calculate_route_length)):
        data = samples[i]
        G = data['G']
        lanes = data['lanes']

        ego_lane_index = np.argmin(np.min(np.linalg.norm(lanes, axis=-1), axis=1))
        
        # find all paths from ego lane to all other lanes
        num_lanes = len(lanes)
        paths = []
        for l1 in range(num_lanes):
            if l1 == ego_lane_index:
                continue
            
            if nx.has_path(G, source=ego_lane_index, target=l1):
                paths.append(nx.shortest_path(G, source=ego_lane_index, target=l1))
        paths.append([ego_lane_index])

        # find the length of each path
        path_lengths = []
        for path in paths:
            assert path[0] == ego_lane_index
            start_idx = np.argmin(np.linalg.norm(lanes[path[0]], axis=-1))
            
            path_length = 0
            for li, lane_index in enumerate(path):
                if li == 0:
                    path_length += get_lane_length(lanes[lane_index, start_idx:])
                else:
                    path_length += get_lane_length(lanes[lane_index])

            path_lengths.append(path_length)
        
        # append the maximum path length
        path_lengths_all.append(max(path_lengths))

    path_lengths_all = np.array(path_lengths_all)

    return np.mean(path_lengths_all), np.std(path_lengths_all, ddof=1)


def compute_endpoint_dist(samples):
    """ Computes the mean/std distance between the end/start points of connecting (suc/pre edge) lanes in the dataset."""
    """ ---l0--->(end) (start)---l1---> compute: dist(end, start) """
    print("Computing endpoint distances")
    num_samples_to_calculate_route_length = len(samples)
    endpoint_distances = []
    for i in tqdm(range(num_samples_to_calculate_route_length)):
        data = samples[i]
        G = data['G']
        lanes = data['lanes']

        edges = G.edges()
        for edge in edges:
            src = edge[0]
            dst = edge[1]

            end = lanes[src, -1]
            start = lanes[dst, 0]

            endpoint_distances.append(np.linalg.norm(end-start))
    
    endpoint_distances = np.array(endpoint_distances)
    return endpoint_distances.mean(), endpoint_distances.std(ddof=1)


def get_keypoint_G(G, lanes):
    """ Constructs a keypoint graph from the lane graph G and the lanes data."""
    keypoint_G = nx.DiGraph()
    for lane in G.nodes():
        start_kp = f'kp_start_{lane}'
        end_kp = f'kp_end_{lane}'
        length = get_lane_length(lanes[lane])
        keypoint_G.add_edge(start_kp, end_kp, length=length)

    equivalent_key_points = {}
    counter = 0
    for edge in G.edges():
        kp_1 = f'kp_end_{edge[0]}'
        kp_2 = f'kp_start_{edge[1]}'
        
        found_in_dict = False
        for kp in equivalent_key_points:
            if kp_1 in equivalent_key_points[kp] or kp_2 in equivalent_key_points[kp]:
                equivalent_key_points[kp].add(kp_1)
                equivalent_key_points[kp].add(kp_2)
                found_in_dict = True
        
        if not found_in_dict:
            new_kp = f'kp_{counter}'
            counter += 1
            kp_1 = f'kp_end_{edge[0]}'
            kp_2 = f'kp_start_{edge[1]}'
            equivalent_key_points[new_kp] = set([kp_1, kp_2])

    inv_equivalent_key_points = {}
    for kp in equivalent_key_points:
        for old_kp in equivalent_key_points[kp]:
            inv_equivalent_key_points[old_kp] = kp
    
    mapping = {}
    for node in keypoint_G.nodes():
        if node in inv_equivalent_key_points:
            mapping[node] = inv_equivalent_key_points[node]
        else:
            mapping[node] = node

    keypoint_G = nx.relabel_nodes(keypoint_G, mapping, copy=True)

    return keypoint_G


def get_num_keypoints(G):
    """ Returns the number of keypoints in the graph G."""
    return np.array([len(G)])


def get_degree_keypoints(G):
    """ Returns the degree of each keypoint in the graph G."""
    degrees = [val for (node, val) in G.degree()]
    return np.array(degrees)


# From SLEDGE Authors
def urban_planning_reach_and_convenience(G_edges):
    """ Computes the reach and convenience metrics based on implementation
    provided by SLEDGE authors."""
    reach = []
    convenience = []

    for source in G_edges.nodes():
        # Compute shortest path lengths from this source to all reachable nodes
        lengths_dict = dict(nx.single_source_shortest_path_length(G_edges, source))
        reach.append(len(lengths_dict) - 1)
        for target, length in lengths_dict.items():
            if source != target:
                convenience.append(length)

    return np.array(reach), np.array(convenience)


def get_onroad_vehicles(vehicles, lanes, tol=1.5):
    """ Filters the vehicles that are on the road based on their distance to the lanes."""
    lanes = lanes.reshape(-1, 2)
    
    vehicle_road_dist = np.linalg.norm(vehicles[:, np.newaxis, :UNIFIED_FORMAT_INDICES['pos_y'] + 1] - lanes[np.newaxis, :, :], axis=-1).min(1)
    offroad_mask = vehicle_road_dist > tol # following SceneControl
    onroad_vehicles = np.where(~offroad_mask)[0]

    return vehicles[onroad_vehicles]


def get_nearest_dists(vehicles):
    """ Computes the nearest distance between vehicles in the scene."""
    vehicle_vehicle_dist = np.linalg.norm(vehicles[:, np.newaxis, :UNIFIED_FORMAT_INDICES['pos_y'] + 1] - vehicles[np.newaxis, :, :UNIFIED_FORMAT_INDICES['pos_y'] + 1], axis=-1)
    # set the distance to self to a large value to avoid self-distance
    for i in range(len(vehicles)):
        vehicle_vehicle_dist[i, i] = 1000

    return vehicle_vehicle_dist.min(1)


def get_lateral_devs(vehicles, lanes):
    """ Computes the lateral deviations of vehicles from the nearest lane."""
    agents_expanded = vehicles[:, np.newaxis, np.newaxis, :UNIFIED_FORMAT_INDICES['pos_y'] + 1]  # Shape (A, 1, 1, 2)
    diffs = agents_expanded - lanes[np.newaxis, :, :, :]
    dists_squared = np.sum(diffs**2, axis=-1)  # Shape (A, N, 20)
    min_dists_squared = np.min(dists_squared, axis=(1, 2))
    
    return np.sqrt(min_dists_squared)


def get_angular_devs(vehicles, lanes):
    """ Computes the angular deviations of vehicles from the nearest lane segment."""
    agent_positions = vehicles[:, :UNIFIED_FORMAT_INDICES['pos_y'] + 1]  # Extract positions (x, y)
    cos_theta = vehicles[:, UNIFIED_FORMAT_INDICES['cos_heading']]
    sin_theta = vehicles[:, UNIFIED_FORMAT_INDICES['sin_heading']]
    agent_headings = np.arctan2(sin_theta, cos_theta)

    agents_expanded = agent_positions[:, np.newaxis, np.newaxis, :]
    direction_vectors = lanes[:, 1:, :] - lanes[:, :-1, :]
    centerline_headings = np.arctan2(direction_vectors[..., 1], direction_vectors[..., 0])  # Shape (N, 19)
    diffs = agents_expanded - lanes[np.newaxis, :, :, :]  # Shape (A, N, 20, 2)
    dists_squared = np.sum(diffs**2, axis=-1)  # Shape (A, N, 20)
    # Find the indices of the nearest centerline point for each agent
    nearest_flat_indices = np.argmin(dists_squared.reshape(dists_squared.shape[0], -1), axis=-1)
    nearest_centerline_indices = nearest_flat_indices // dists_squared.shape[2]
    nearest_point_indices = nearest_flat_indices % dists_squared.shape[2]

    # Handle the case where nearest point is the first or last in the centerline
    nearest_point_indices = np.clip(nearest_point_indices, 1, dists_squared.shape[-1] - 1)
    # Get the corresponding headings of the nearest segments
    nearest_centerline_headings = centerline_headings[nearest_centerline_indices, nearest_point_indices - 1]
    
    # Compute the angular deviation in radians and convert to degrees
    angular_deviation_radians = np.arctan2(np.sin(agent_headings - nearest_centerline_headings),
                                           np.cos(agent_headings - nearest_centerline_headings))  # Ensure correct angle difference
    angular_deviation_degrees = np.degrees(angular_deviation_radians)

    return angular_deviation_degrees


def get_lengths(vehicles):
    """ Returns the lengths of the vehicles in the scene."""
    return vehicles[:, UNIFIED_FORMAT_INDICES['length']]


def get_widths(vehicles):
    """ Returns the widths of the vehicles in the scene."""
    return vehicles[:, UNIFIED_FORMAT_INDICES['width']]


def get_speeds(vehicles):
    """ Returns the speeds of the vehicles in the scene."""
    return vehicles[:, UNIFIED_FORMAT_INDICES['speed']]


def compute_urban_planning_metrics(samples, gt_samples):
    """ Computes the urban planning metrics for the samples and ground truth samples."""
    print("Computing urban planning frechet distances")
    
    # gen = scenario dreamer samples, real = gt samples
    degree_keypoints_gen_all = []
    num_keypoints_gen_all = []
    num_valid_paths_gen_all = []
    valid_path_lengths_gen_all = []
    degree_keypoints_real_all = []
    num_keypoints_real_all = []
    num_valid_paths_real_all = []
    valid_path_lengths_real_all = []
    for i in tqdm(range(len(samples))):
        data_gen = samples[i]
        data_real = gt_samples[i]

        G_gen = data_gen['G']
        lanes_gen = data_gen['lanes']
        G_real = data_real['G']
        lanes_real = data_real['lanes']

        keypoint_G_gen = get_keypoint_G(G_gen, lanes_gen)
        keypoint_G_real = get_keypoint_G(G_real, lanes_real)

        # get statistics for generated scene
        degree_keypoints_gen = get_degree_keypoints(keypoint_G_gen)
        num_keypoints_gen = get_num_keypoints(keypoint_G_gen)
        num_valid_paths_gen, valid_path_lengths_gen = urban_planning_reach_and_convenience(keypoint_G_gen)
        degree_keypoints_gen_all.append(degree_keypoints_gen)
        num_keypoints_gen_all.append(num_keypoints_gen)
        num_valid_paths_gen_all.append(num_valid_paths_gen)
        valid_path_lengths_gen_all.append(valid_path_lengths_gen)

        # get statistics for real scene
        degree_keypoints_real = get_degree_keypoints(keypoint_G_real)
        num_keypoints_real = get_num_keypoints(keypoint_G_real)
        num_valid_paths_real, valid_path_lengths_real = urban_planning_reach_and_convenience(keypoint_G_real)
        degree_keypoints_real_all.append(degree_keypoints_real)
        num_keypoints_real_all.append(num_keypoints_real)
        num_valid_paths_real_all.append(num_valid_paths_real)
        valid_path_lengths_real_all.append(valid_path_lengths_real)

    degree_keypoints_gen_all = np.concatenate(degree_keypoints_gen_all, axis=0)
    num_keypoints_gen_all = np.concatenate(num_keypoints_gen_all, axis=0)
    num_valid_paths_gen_all = np.concatenate(num_valid_paths_gen_all, axis=0)
    valid_path_lengths_gen_all = np.concatenate(valid_path_lengths_gen_all, axis=0)

    degree_keypoints_real_all = np.concatenate(degree_keypoints_real_all, axis=0)
    num_keypoints_real_all = np.concatenate(num_keypoints_real_all, axis=0)
    num_valid_paths_real_all = np.concatenate(num_valid_paths_real_all, axis=0)
    valid_path_lengths_real_all = np.concatenate(valid_path_lengths_real_all, axis=0)

    frechet_connectivity = compute_frechet_distance(degree_keypoints_gen_all, degree_keypoints_real_all) * 10
    frechet_density = compute_frechet_distance(num_keypoints_gen_all, num_keypoints_real_all)
    frechet_reach = compute_frechet_distance(num_valid_paths_gen_all, num_valid_paths_real_all)
    frechet_convenience = compute_frechet_distance(valid_path_lengths_gen_all, valid_path_lengths_real_all) * 10

    return frechet_connectivity, frechet_density, frechet_reach, frechet_convenience


def compute_jsd_metrics(samples, gt_samples):
    """ Computes the JSD agent metrics for the samples and ground truth samples."""
    print("Computing agent jsd metrics")
    nearest_dist_gen_all = []
    lat_dev_gen_all = []
    ang_dev_gen_all = []
    length_gen_all = []
    width_gen_all = []
    speed_gen_all = []
    nearest_dist_real_all = []
    lat_dev_real_all = []
    ang_dev_real_all = []
    length_real_all = []
    width_real_all = []
    speed_real_all = []

    for i in tqdm(range(len(samples))):
        data_gen = samples[i]
        vehicles_gen = data_gen['vehicles'] # [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
        # resample lanes to higher resolution
        lanes_gen = resample_lanes(data_gen['lanes'], num_points=100)
        onroad_vehicles_gen = get_onroad_vehicles(vehicles_gen, lanes_gen)

        if len(vehicles_gen) > 1:
            nearest_dist_gen_all.append(get_nearest_dists(vehicles_gen))
        if len(onroad_vehicles_gen) > 0:
            lat_dev_gen_all.append(get_lateral_devs(onroad_vehicles_gen, lanes_gen))
            ang_dev_gen_all.append(get_angular_devs(onroad_vehicles_gen, lanes_gen))
        length_gen_all.append(get_lengths(vehicles_gen))
        width_gen_all.append(get_widths(vehicles_gen))
        speed_gen_all.append(get_speeds(vehicles_gen))

        data_real = gt_samples[i]
        vehicles_real = data_real['vehicles'] # [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
        
        lanes_real = resample_lanes(data_real['lanes'], num_points=100)
        onroad_vehicles_real = get_onroad_vehicles(vehicles_real, lanes_real)

        if len(vehicles_real) > 1:
            nearest_dist_real_all.append(get_nearest_dists(vehicles_real))
        if len(onroad_vehicles_real) > 0:
            lat_dev_real_all.append(get_lateral_devs(onroad_vehicles_real, lanes_real))
            ang_dev_real_all.append(get_angular_devs(onroad_vehicles_real, lanes_real))
        length_real_all.append(get_lengths(vehicles_real))
        width_real_all.append(get_widths(vehicles_real))
        speed_real_all.append(get_speeds(vehicles_real))

    nearest_dist_gen_all = np.concatenate(nearest_dist_gen_all, axis=0)
    lat_dev_gen_all = np.concatenate(lat_dev_gen_all, axis=0)
    ang_dev_gen_all = np.concatenate(ang_dev_gen_all, axis=0)
    length_gen_all = np.concatenate(length_gen_all, axis=0)
    width_gen_all = np.concatenate(width_gen_all, axis=0)
    speed_gen_all = np.concatenate(speed_gen_all, axis=0)

    nearest_dist_real_all = np.concatenate(nearest_dist_real_all, axis=0)
    lat_dev_real_all = np.concatenate(lat_dev_real_all, axis=0)
    ang_dev_real_all = np.concatenate(ang_dev_real_all, axis=0)
    length_real_all = np.concatenate(length_real_all, axis=0)
    width_real_all = np.concatenate(width_real_all, axis=0)
    speed_real_all = np.concatenate(speed_real_all, axis=0)

    nearest_dist_jsd = jsd(nearest_dist_gen_all, nearest_dist_real_all, clip_min=0, clip_max=50, bin_size=1) * 10
    lat_dev_jsd = jsd(lat_dev_gen_all, lat_dev_real_all, clip_min=0, clip_max=1.5, bin_size=0.1) * 10
    ang_dev_jsd = jsd(ang_dev_gen_all, ang_dev_real_all, clip_min=-200, clip_max=200, bin_size=5) * 100
    length_jsd = jsd(length_gen_all, length_real_all, clip_min=0, clip_max=25, bin_size=0.1) * 100
    width_jsd = jsd(width_gen_all, width_real_all, clip_min=0, clip_max=5, bin_size=0.1) * 100
    speed_jsd = jsd(speed_gen_all, speed_real_all, clip_min=0, clip_max=50, bin_size=1) * 100

    return nearest_dist_jsd, lat_dev_jsd, ang_dev_jsd, length_jsd, width_jsd, speed_jsd


def compute_lane_metrics(samples, gt_samples):
    """ Computes the lane metrics for the samples and ground truth samples."""
    # later release
    # fd, precision, recall = compute_perceptual_metrics(samples, gt_samples, model=model)
    route_length_mean, route_length_std = compute_route_length(samples)
    endpoint_dist_mean, endpoint_dist_std = compute_endpoint_dist(samples)
    frechet_connectivity, frechet_density, frechet_reach, frechet_convenience = compute_urban_planning_metrics(samples, gt_samples)

    return {
        'route_length_mean': route_length_mean,
        'route_length_std': route_length_std,
        'endpoint_dist_mean': endpoint_dist_mean,
        'endpoint_dist_std': endpoint_dist_std,
        'frechet_connectivity': frechet_connectivity,
        'frechet_density': frechet_density,
        'frechet_reach': frechet_reach,
        'frechet_convenience': frechet_convenience,
    }

def compute_agent_metrics(samples, gt_samples):
    """ Computes the agent metrics for the samples and ground truth samples."""
    nearest_dist_jsd, lat_dev_jsd, ang_dev_jsd, length_jsd, width_jsd, speed_jsd = compute_jsd_metrics(samples, gt_samples)
    collision_rate = compute_collision_rate(samples)

    return {
        'nearest_dist_jsd': nearest_dist_jsd,
        'lat_dev_jsd': lat_dev_jsd,
        'ang_dev_jsd': ang_dev_jsd,
        'length_jsd': length_jsd,
        'width_jsd': width_jsd,
        'speed_jsd': speed_jsd,
        'collision_rate': collision_rate * 100
    }