import math
import os
import copy
import pickle
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from waymo_open_dataset.protos import scenario_pb2
from google.protobuf.json_format import MessageToDict
from cfgs.config import CONFIG_PATH

ERR_VAL = -1e4
_WAYMO_OBJECT_STR = {
    'TYPE_UNSET': "unset",
    'TYPE_VEHICLE': "vehicle",
    'TYPE_PEDESTRIAN': "pedestrian",
    'TYPE_CYCLIST': "cyclist",
    'TYPE_OTHER': "other",
}

def poly_gon_and_line(poly_dict):

    plg_xyz = []

    if type(poly_dict) == list:
        for plg in poly_dict:
            plg_xyz += [[plg['x'],plg['y'],plg['z']]]
    else:
        plg_xyz = [poly_dict['x'],poly_dict['y'],poly_dict['z']]

    plg_xyz = np.array(plg_xyz)

    return plg_xyz


def road_info_except_lane(x_list, road_keys):

    output = {}
    output['id'] = []
    
    key_x = list(x_list[0].keys())[1]
    keys = road_keys[key_x]
    for key in keys:
        output[key] = []

    for x in x_list:
        output['id'] += [x['id']]
        for key in keys:
            if key in list(x[key_x].keys()):
                if key[0] == 'p':
                    output[key] += [poly_gon_and_line(x[key_x][key])]
                else:
                    output[key] += [x[key_x][key]]
            else:
                output[key] += [None]
    
    return output


def road_info_lane(x_dict):
    
    lanes = dict()

    for ln in x_dict:
        
        ln_info = dict()
        ln_id = ln['id']

        for key in ln['lane'].keys():
            if key[0] == 'p':
                ln_info[key] = poly_gon_and_line(ln['lane']['polyline'])
            else:
                ln_info[key] = ln['lane'][key]

        lanes[ln_id] = ln_info
    
    return lanes


def get_lane_pairs(engage_lanes):

    lane_ids = list(engage_lanes.keys())
    pre_pairs, suc_pairs = {}, {}
    left_pairs, right_pairs = {}, {}

    for i, lane_id in enumerate(lane_ids):

        lane = engage_lanes[lane_id]

        if 'entryLanes' in lane.keys():
            for eL in lane['entryLanes']:
                if eL in lane_ids:
                    if int(lane_id) in pre_pairs:
                        pre_pairs[int(lane_id)].append(int(eL))
                    else:
                        pre_pairs[int(lane_id)] = [int(eL)]
        
        if 'exitLanes' in lane.keys():
            for eL in lane['exitLanes']:
                if eL in lane_ids:
                    if int(lane_id) in suc_pairs:
                        suc_pairs[int(lane_id)].append(int(eL))
                    else:
                        suc_pairs[int(lane_id)] = [int(eL)]

        if 'leftNeighbors' in lane.keys():
            for left in lane['leftNeighbors']:
                left = left['featureId']
                if left in lane_ids:
                    if int(lane_id) in left_pairs:
                        left_pairs[int(lane_id)].append(int(left))
                    else:
                        left_pairs[int(lane_id)] = [int(left)]

        # Add right neighbors
        if 'rightNeighbors' in lane.keys():
            for right in lane['rightNeighbors']:
                right = right['featureId']
                if right in lane_ids:
                    if int(lane_id) in right_pairs:
                        right_pairs[int(lane_id)].append(int(right))
                    else:
                        right_pairs[int(lane_id)] = [int(right)]

    return pre_pairs, suc_pairs, left_pairs, right_pairs

def get_engage_lanes(data):

    lanes = data['road_info']['lane']
    engage_lanes = dict()
    lane_ids = list(lanes.keys())

    for id in lane_ids:
        lane = lanes[id]
        if len(lane['polyline']) < 2: #rule out those 1 point lane
            continue
        else:
            lane = copy.deepcopy(lane)
            engage_lanes[id] = lane
            
    return engage_lanes

def get_lane_graph(data):
    engage_lanes = get_engage_lanes(data)
    pre_pairs, suc_pairs, left_pairs, right_pairs = get_lane_pairs(engage_lanes)
    graph = dict()

    graph['pre_pairs'] = pre_pairs   
    graph['suc_pairs'] = suc_pairs
    graph['left_pairs'] = left_pairs
    graph['right_pairs'] = right_pairs  

    return graph


def process_lanegraph(data):
    """lanes are
    {
        xys: n x 2 array (xy locations)
        in_edges: n x X list of lists
        out_edges: n x X list of lists
        edges: m x 5 (x,y,hcos,hsin,l)
        edgeixes: m x 2 (v0, v1)
        ee2ix: dict (v0, v1) -> ei
    }
    """
    lanes = {}
    pre_pairs = {}
    suc_pairs = {}
    left_pairs = {}
    right_pairs = {}
    for lane_id in data['road_info']['lane']:
        lane_type = data['road_info']['lane'][lane_id]['type']
        if lane_type == 'TYPE_UNDEFINED' or lane_type == 'TYPE_BIKE_LANE':
            continue
        
        my_lane = data['road_info']['lane'][lane_id]['polyline']
        lanes[int(lane_id)] = my_lane[:, :2]
        
        if int(lane_id) in data['graph']['pre_pairs'].keys():
            pre_pairs[int(lane_id)] = data['graph']['pre_pairs'][int(lane_id)]
        if int(lane_id) in data['graph']['suc_pairs'].keys():
            suc_pairs[int(lane_id)] = data['graph']['suc_pairs'][int(lane_id)]
        if int(lane_id) in data['graph']['left_pairs'].keys():
            left_pairs[int(lane_id)] = data['graph']['left_pairs'][int(lane_id)]
        if int(lane_id) in data['graph']['right_pairs'].keys():
            right_pairs[int(lane_id)] = data['graph']['right_pairs'][int(lane_id)]

    road_edges = {}
    crosswalks = {}
    stop_signs = {}
    if 'roadEdge' in data['road_info']:
        for i in range(len(data['road_info']['roadEdge']['id'])):
            road_edges[int(data['road_info']['roadEdge']['id'][i])] = data['road_info']['roadEdge']['polyline'][i][:, :2]
    if 'crosswalk' in data['road_info']:
        for i in range(len(data['road_info']['crosswalk']['id'])):
            crosswalks[int(data['road_info']['crosswalk']['id'][i])] = data['road_info']['crosswalk']['polygon'][i][:, :2]
    if 'stopSign' in data['road_info']:
        for i in range(len(data['road_info']['stopSign']['id'])):
            stop_signs[int(data['road_info']['stopSign']['id'][i])] = data['road_info']['stopSign']['position'][i][:2]

    return {'lanes': lanes, 
            'pre_pairs': pre_pairs, 
            'suc_pairs': suc_pairs, 
            'left_pairs': left_pairs, 
            'right_pairs': right_pairs,
            'road_edges': road_edges,
            'crosswalks': crosswalks,
            'stop_signs': stop_signs}


def _parse_object_state(states, final_state):
    return {
        "position": [{
            "x": state['centerX'],
            "y": state['centerY']
        } if state['valid'] else {
            "x": ERR_VAL,
            "y": ERR_VAL
        } for state in states],
        "width": final_state['width'],
        "length": final_state['length'],
        "heading": [
            math.degrees(state['heading']) if state['valid'] else ERR_VAL
            for state in states
        ],  # Use rad here?
        "velocity": [{
            "x": state['velocityX'],
            "y": state['velocityY']
        } if state['valid'] else {
            "x": ERR_VAL,
            "y": ERR_VAL
        } for state in states],
        "valid": [state['valid'] for state in states]
    }


def _init_object(track):
    """Construct a dict representing the state of the object (vehicle, cyclist, pedestrian).

    Args:
        track (scenario_pb2.Track): protobuf representing the scenario

    Returns
    -------
        Optional[Dict[str, Any]]: dict representing the trajectory and velocity of an object.
    """
    
    final_valid_index = 0
    for i, state in enumerate(track['states']):
        if state['valid']:
            final_valid_index = i

    # not a car
    if 'width' not in track['states'][final_valid_index]:
        return None

    obj = _parse_object_state(track['states'], track['states'][final_valid_index])
    obj["type"] = _WAYMO_OBJECT_STR[track['objectType']]
    return obj


def get_objects(scenario_list, index):
    objects = []
    av_objects_idx = -1
    
    scen = scenario_list[index]
    av_idx = scen['sdcTrackIndex']
    for i, track in enumerate(scen['tracks']):
        if i == av_idx:
            av_objects_idx = len(objects)
        
        obj_to_append = _init_object(track)
        if obj_to_append is not None:
            objects.append(obj_to_append)

    assert av_objects_idx != -1
    
    return objects, av_objects_idx


# we only retrieve lanes, as we are only generating the lane graph
def get_road_info(scenario_list, index):
    scen = scenario_list[index]
    map_feature = dict()

    road_keys = dict()

    road_keys['crosswalk'] = ['polygon']
    road_keys['stopSign'] = ['position']
    road_keys['roadEdge'] = ['polyline']

    if 'mapFeatures' not in scen:
        return None
    
    for mf in scen['mapFeatures']:
        key = list(mf.keys())[1]
        if key in map_feature.keys():
            map_feature[key] += [mf]
        else:
            map_feature[key] = [mf]
    
    road_info = dict()
    for key in map_feature.keys():
        if key == 'lane':
            road_info[key] = road_info_lane(map_feature[key]) 
        elif key in ['roadEdge', 'crosswalk', 'stopSign']:
            road_info[key] = road_info_except_lane(map_feature[key],road_keys)  
        
    return road_info


def collect_data(cfg, output_path, files_path, files, chunk):
    for c in tqdm(chunk):
        filename_path = os.path.join(files_path, files[c])
        dataset = tf.data.TFRecordDataset(filename_path, compression_type='')
        scenario_list = []    
        for data in dataset:
            proto_string = data.numpy()
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(proto_string)
            scenario_dict = MessageToDict(proto)
            scenario_list += [scenario_dict]
        
        for i in range(len(scenario_list)):
            output_file = f'{files[c]}_{i}.pkl'
        
            data = {}
            road_info = get_road_info(scenario_list, i)
            if road_info is None:
                continue
            data['road_info'] = road_info
            
            if 'lane' not in data['road_info']:
                continue
            
            data['graph'] = get_lane_graph(data)
            
            scenario = {}
            lane_graph = process_lanegraph(data)
            objects, av_idx = get_objects(scenario_list, i)

            ### VISUALIZATION FOR TESTING PURPOSES
            # print("Visualizing", i)
            # for lane_id in lane_graph['lanes'].keys():
            #     to_plot = lane_graph['lanes'][lane_id]
            #     plt.plot(to_plot[:, 0], to_plot[:, 1], color='grey', linewidth=0.5)
            #     idx = len(to_plot) // 2
            #     plt.annotate(lane_id,
            #         (to_plot[idx, 0], to_plot[idx, 1]), zorder=5, fontsize=1, color='blue')

            # for road_edge_id in lane_graph['road_edges'].keys():
            #     to_plot = lane_graph['road_edges'][road_edge_id]
            #     plt.plot(to_plot[:, 0], to_plot[:, 1], linewidth=0.75)
            #     idx = len(to_plot) // 2
            #     plt.annotate(road_edge_id,
            #         (to_plot[idx, 0], to_plot[idx, 1]), zorder=5, fontsize=4, color='red')

            # for stop_sign_id in lane_graph['stop_signs'].keys():
            #     to_plot = lane_graph['stop_signs'][stop_sign_id]
            #     plt.scatter(to_plot[0], to_plot[1], color='red', s=10)
            
            # for crosswalk_id in lane_graph['crosswalks'].keys():
            #     to_plot = lane_graph['crosswalks'][crosswalk_id]
            #     plt.plot(to_plot[:, 0], to_plot[:, 1], color='green', linewidth=0.5)
            
            # plt.gca().set_aspect('equal')
            # plt.savefig(f'lane_graph_{i}.png', dpi=1000)
            # plt.clf()

            scenario['lane_graph'] = lane_graph
            scenario['objects'] = objects
            scenario['av_idx'] = av_idx 

            scenario_path = os.path.join(output_path, output_file)
            with open(scenario_path, "wb") as f:
                pickle.dump(scenario, f)


def _work_one_chunk(idx, cfg_dict):
    """Helper so the Pool can pickle the cfg."""
    # Re-create Hydra config in the subprocess
    cfg = OmegaConf.create(cfg_dict)
    cfg.generate_waymo_dataset.chunk_idx = idx
    _run_one_cfg(cfg)                 # <-- see wrapper below


def _run_one_cfg(cfg):
    """A tiny wrapper around your existing logic."""
    if cfg.generate_waymo_dataset.mode == 'train':
        files_path = cfg.waymo_train_folder
        output_path = cfg.generate_waymo_dataset.output_data_folder_train
    elif cfg.generate_waymo_dataset.mode == 'val':
        files_path = cfg.waymo_val_folder
        output_path = cfg.generate_waymo_dataset.output_data_folder_val
    else:
        files_path = cfg.waymo_test_folder
        output_path = cfg.generate_waymo_dataset.output_data_folder_test

    files = sorted(os.listdir(files_path))
    start = cfg.generate_waymo_dataset.chunk_idx * cfg.generate_waymo_dataset.chunk_size
    end   = start + cfg.generate_waymo_dataset.chunk_size
    chunk = [i for i in range(start, min(end, len(files)))]
    if not chunk:
        return

    os.makedirs(output_path, exist_ok=True)
    collect_data(cfg, output_path, files_path, files, chunk)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    """
    If chunk_idx >= 0 → behave exactly as before (single chunk).
    If chunk_idx  < 0 → farm out *all* chunks to a worker pool.
    """
    if cfg.generate_waymo_dataset.chunk_idx >= 0:
        _run_one_cfg(cfg)
        return

    # ---  fan out  ---
    if cfg.generate_waymo_dataset.mode == 'train':
        total_chunks = 20
    else:               # val or test
        total_chunks = 5

    n_workers = min(cfg.generate_waymo_dataset.get("num_workers", mp.cpu_count()),
                    total_chunks)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(_work_one_chunk,
                     [(i, cfg_dict) for i in range(total_chunks)])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()