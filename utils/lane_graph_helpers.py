import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx

def find_lane_groups(pre_pairs, suc_pairs):
    """Group lane IDs into compact lanes based on lane compression algorithm originally described here: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00490-supp.pdf"""
    def dfs(lane_id, group):
        visited.add(lane_id)
        group.append(lane_id)
        
        if len(suc_pairs[lane_id]) == 1:
            next_lane = suc_pairs[lane_id][0]
            if next_lane not in visited and len(pre_pairs[next_lane]) == 1:
                dfs(next_lane, group)

    lane_groups = []
    visited = set()

    def find_starting_lanes(pre_pairs, suc_pairs):
        starting_lanes = set()
        all_lanes = set(pre_pairs.keys()).union(set(suc_pairs.keys()))
        
        for lane_id in all_lanes:
            if len(pre_pairs[lane_id]) == 1:
                if len(suc_pairs[pre_pairs[lane_id][0]]) == 1:
                    continue
            
            starting_lanes.add(lane_id)
        
        return starting_lanes

    starting_lanes = find_starting_lanes(pre_pairs, suc_pairs)

    for lane_id in starting_lanes:
        if lane_id not in visited:
            group = []
            dfs(lane_id, group)
            if group:
                lane_groups.append(group)

    # edge case: get the centerline cycles that are fully closed (all degree 2 lane segments)
    num_lane_ids = sum([len(lane_groups[i]) for i in range(len(lane_groups))])
    while num_lane_ids != len(pre_pairs.keys()):
        unaccounted_ids = list(set(list(pre_pairs.keys())).difference(set([x for xs in lane_groups for x in xs])))
        
        lane_id = unaccounted_ids[0]
        if lane_id not in visited:
            group = []
            dfs(lane_id, group)
            if group:
                lane_groups.append(group)

        num_lane_ids = sum([len(lane_groups[i]) for i in range(len(lane_groups))])

    lane_groups_dict = {}
    for i in range(len(lane_groups)):
        lane_groups_dict[i] = lane_groups[i]

    return lane_groups_dict


def find_lane_group_id(lane_id, lane_groups):
    """Return the ID of the lane-group containing `lane_id`"""
    for lane_group_id in lane_groups:
        if lane_id in lane_groups[lane_group_id]:
            return lane_group_id
        

def resample_polyline(points, num_points=20):
    """Resample a polyline to `num_points` equally spaced points along its arc-length."""
    # Calculate the cumulative distances along the polyline
    distances = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
    # Create an array of 20 evenly spaced distance values along the polyline
    target_distances = np.linspace(0, cumulative_distances[-1], num=num_points)
    
    # Interpolate to find x and y values at these target distances
    x_new = np.interp(target_distances, cumulative_distances, points[:, 0])
    y_new = np.interp(target_distances, cumulative_distances, points[:, 1])
    
    # Combine x and y coordinates into a single array
    new_points = np.stack((x_new, y_new), axis=-1)
    
    return new_points


def resample_lanes(lanes, num_points):
    """Resample a list of lanes (each lane is a polyline) to have `num_points` equally spaced points along each lane's arc-length."""
    lanes_resampled = []
    for lane in lanes:
        lanes_resampled.append(resample_polyline(lane, num_points=num_points))

    return np.array(lanes_resampled)


def adjacency_matrix_to_adjacency_list(lane_graph_adj):
    """Convert an adjacency matrix to an adjacency list representation."""
    num_lanes = len(lane_graph_adj)
    
    G = nx.DiGraph(incoming_graph_data=lane_graph_adj)
    pre_pairs = {}
    suc_pairs = {}
    for lane_id in range(num_lanes):
        pre_pairs[lane_id] = []
        suc_pairs[lane_id] = []

    for edge in G.edges():
        pre_pairs[edge[1]].append(edge[0])
        suc_pairs[edge[0]].append(edge[1])
    
    return pre_pairs, suc_pairs