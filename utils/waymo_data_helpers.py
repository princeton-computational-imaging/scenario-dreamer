import numpy as np
import torch
np.set_printoptions(suppress=True)
from utils.geometry import apply_se2_transform
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage
import networkx as nx
import copy
import random
from tqdm import tqdm

def get_object_type_onehot(agent_type):
    """Return the one-hot NumPy vector encoding of an agent type."""
    agent_types = {"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
    return np.eye(len(agent_types))[agent_types[agent_type]]

def get_lane_connection_type_onehot(lane_connection_type):
    """Return the one-hot NumPy vector encoding of a lane-connection type."""
    lane_connection_types = {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}
    return np.eye(len(lane_connection_types))[lane_connection_types[lane_connection_type]]