import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx

def get_compact_lane_graph(data):
    """Apply lane graph compression algorithm (merging lanes that connect with node degree=2).

    The resulting compact graph uses *lane-group* identifiers where
    contiguous segments have been concatenated.  All
    connection dictionaries (``pre``, ``succ``, ``left``, ``right``)
    are updated to reference the new identifiers.

    Parameters
    ----------
    data
        Dictionary from a raw Waymo pickle.  Requires the key
        ``"lane_graph"``.

    Returns
    -------
    compact_lane_graph
        A dict with the same layout as the original Waymo lane graph
        but using merged lanes.
    """
    
    lane_ids = data['lane_graph']['lanes'].keys()
    pre_pairs = data['lane_graph']['pre_pairs']
    suc_pairs = data['lane_graph']['suc_pairs']
    left_pairs = data['lane_graph']['left_pairs']
    right_pairs = data['lane_graph']['right_pairs']

    # Remove dangling references ------------------------------------
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

    # Ensure every lane appears as a key ----------------------------
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
            # first lane in group is used to find predecessor lane group
            if i == 0:
                compact_lane.append(data['lane_graph']['lanes'][lane_id])
                
                if len(pre_pairs[lane_id]) > 0:
                    for pre_lane_id in pre_pairs[lane_id]:
                        compact_pre_pair.append(find_lane_group_id(pre_lane_id, lane_groups))
            else:
                # avoid duplicate coordinates
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

            # last lane in group is used to find successor lane group
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
    
    compact_lane_graph = {
        'lanes': compact_lanes,
        'pre_pairs': compact_pre_pairs,
        'suc_pairs': compact_suc_pairs,
        'left_pairs': compact_left_pairs,
        'right_pairs': compact_right_pairs
    }

    return compact_lane_graph


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


def resample_polyline_every(polyline, every=1.5):
    """ Resample a polyline to have points spaced every `every` m along its arc-length."""
    # Calculate the distance between each consecutive pair of points
    distances = np.sqrt(((np.diff(polyline, axis=0)) ** 2).sum(axis=1))
    
    # Calculate cumulative distance along the polyline
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
    # Determine the new distances at which to sample points
    target_distances = np.arange(0, cumulative_distances[-1], every)
    
    # Interpolate the x and y coordinates at the target distances
    resampled_x = np.interp(target_distances, cumulative_distances, polyline[:, 0])
    resampled_y = np.interp(target_distances, cumulative_distances, polyline[:, 1])
    
    # Stack the x and y coordinates to form the resampled polyline
    resampled_polyline = np.vstack((resampled_x, resampled_y)).T
    
    return resampled_polyline


def resample_lanes(lanes, num_points):
    """Resample a list of lanes (each lane is a polyline) to have `num_points` equally spaced points along each lane's arc-length."""
    lanes_resampled = []
    for lane in lanes:
        lanes_resampled.append(resample_polyline(lane, num_points=num_points))

    return np.array(lanes_resampled)


def resample_lanes_with_mask(lanes, lanes_mask, num_points):
    """Resample a list of lanes (each lane is a polyline) to have `num_points` 
    equally spaced points along each lane's arc-length, using a mask to indicate valid lanes."""
    lanes_resampled = []
    for lane, mask in zip(lanes, lanes_mask):
        if mask.sum():
            lanes_resampled.append(resample_polyline(lane[mask], num_points))
    
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