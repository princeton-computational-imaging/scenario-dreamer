import os
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm
import hydra
from cfgs.config import CONFIG_PATH

ROAD_EDGE_OFFSET = 4.83  # meters laterally offset from the route

def reverse_ag_type_mapping(agent_type_onehot):
    """ Reverse the agent type mapping from one-hot to string. """
    agent_types = {0: "unset", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}
    ag_type_idx = agent_type_onehot.argmax()
    return agent_types[ag_type_idx]


def compute_route_road_edges(route):
    """ Compute the road edges from the route. """
    repeated_route = np.vstack((route[0], route, route[-1]))
    road_edges = np.zeros((2, *route.shape))
    diffs = repeated_route[2:] - repeated_route[:-2]
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    unit_directions = diffs / norms

    # Get perpendicular directions by rotating 90 degrees
    perp_directions = np.stack([-unit_directions[:, 1], unit_directions[:, 0]], axis=1)

    # Add offsets to the route points to create corridors
    right_corridor = repeated_route[1:-1] + perp_directions * ROAD_EDGE_OFFSET
    left_corridor = repeated_route[1:-1] - perp_directions * ROAD_EDGE_OFFSET
    road_edges[0] = left_corridor
    road_edges[1] = right_corridor

    return road_edges


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    path_to_pickles = cfg.convert_pickles_to_jsons.path_to_pickles
    path_to_jsons = cfg.convert_pickles_to_jsons.path_to_jsons
    dataset_size = cfg.convert_pickles_to_jsons.dataset_size

    pkl_fnames = glob.glob(f"{path_to_pickles}/*.pkl")
    print("Number of scenes", len(pkl_fnames))
    num_converted_files = 0

    os.makedirs(path_to_jsons, exist_ok=True)
    for raw_path in tqdm(pkl_fnames):
        with open(raw_path, 'rb') as f:
            data = pickle.load(f)
        
        new_data = {}
        # store metadata
        new_data['tl_states'] = {}
        new_fname = raw_path.split("/")[-1].split("validation.")[-1].replace("pkl", "json")
        new_data['name'] = new_fname
        # Add scenario_id (filename without extension)
        new_data['scenario_id'] = new_fname.replace('.json', '')
        if 'ego_index' not in data:
            data['ego_index'] = len(data['agents']) - 1
        new_data['ego_idx'] = data['ego_index']
        
        # store route
        curr_route =  [{'x': data['route'][i, 0], 
                        'y': data['route'][i, 1]
                        } for i in range(len(data['route']))]
        assert len(curr_route) > 0
        new_data['route'] = curr_route
        
        # store lane centerlines
        new_data['roads'] = []
        for s in range(len(data['lanes'])):
            curr_road_pts = [{
                'x': data['lanes'][s, i, 0], 
                'y': data['lanes'][s, i, 1]
            } for i in range(data['lanes'][s].shape[0])]
            curr_road_dict = {'geometry': curr_road_pts, 
                            'type': 'lane'}
            new_data['roads'].append(curr_road_dict)

        # compute and store road edges defining the corridor around the route 
        road_edges = compute_route_road_edges(route=data['route'])
        for s in range(len(road_edges)):
            curr_road_pts = [{'x': road_edges[s, i, 0], 
                              'y': road_edges[s, i, 1]
                              } for i in range(road_edges[s].shape[0])]
            curr_road_dict = {'geometry': curr_road_pts, 'type': 'road_edge'}
            new_data['roads'].append(curr_road_dict)

        # store objects
        new_data['objects'] = []
        for n in range(len(data['agents'])):
            positions  = data['agents'][n, :, :2]
            positions  = [{'x': positions[i, 0], 
                           'y': positions[i, 1]
                           } for i in range(len(positions))]
            velocities = data['agents'][n, :, 2:4]
            velocities = [{'x': velocities[i, 0], 
                           'y': velocities[i, 1]
                           } for i in range(len(velocities))]
            headings   = data['agents'][n, :, 4]
            length     = data['agents'][n, :, 5]
            width      = data['agents'][n, :, 6]
            valid      = data['agents'][n, :, 7].astype(bool)
            goals      = data['agents'][n, valid, :2][-1]
            goals      = {'x': goals[0], 'y': goals[1]}
            ag_type    = reverse_ag_type_mapping(
                            data['agent_types'][n]
                        )
            if n == data['ego_index']:
                mark_as_expert = False
                goals = curr_route[-1]
            else:
                mark_as_expert = True

            height = 1.5
            curr_obj_dict = {
                'position': positions,
                'width': width[valid][0],
                'length': length[valid][0],
                'height': height,
                'id': n,  # Use original index as ID (will be reassigned after reordering)
                'heading': headings.tolist(),
                'velocity': velocities,
                'valid': valid.tolist(),
                'goalPosition': goals,
                'type': ag_type,
                'mark_as_expert': mark_as_expert
            }
            new_data['objects'].append(curr_obj_dict)

        # store ego vehicle as the first object
        new_data['objects'] = [new_data['objects'][data['ego_index']]] + new_data['objects']
        new_data['objects'].pop(data['ego_index']+1)
        
        # Reassign IDs after reordering (ego is now at index 0, others follow sequentially)
        for idx, obj in enumerate(new_data['objects']):
            obj['id'] = idx
        
        # Create metadata with sdc_track_index, tracks_to_predict, and objects_of_interest
        # Since ego is moved to index 0, sdc_track_index should be 0
        sdc_track_index = 0
        
        # Include ego and a few other vehicles in tracks_to_predict
        # Find vehicle indices (excluding ego at index 0)
        vehicle_indices = [0]  # Always include ego
        
        tracks_to_predict = [
            {'track_index': idx, 'difficulty': 0} 
            for idx in vehicle_indices
        ]
        
        # Objects of interest: include ego (ID 0)
        objects_of_interest = [0]
        
        new_data['metadata'] = {
            'sdc_track_index': sdc_track_index,
            'tracks_to_predict': tracks_to_predict,
            'objects_of_interest': objects_of_interest
        }
        
        # store the json
        with open(os.path.join(path_to_jsons, new_fname), 'w') as f:
            json.dump(new_data, f, indent=4)
        
        num_converted_files += 1
        if num_converted_files >= dataset_size:
            break
    
    print(f"GPUDrive dataset size: {num_converted_files}")


if __name__ == "__main__":
    main()
