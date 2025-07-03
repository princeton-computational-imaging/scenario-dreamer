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

class ScenarioDreamerData(HeteroData):
    """Torch-Geometric `HeteroData` that auto-increments agent↔lane edge indices for proper batching."""
    def __inc__(self, key, value, store):
        if 'edge_index' in key:
            # Increment the edge indices based on the number of nodes in the source node type
            if key[0] == 'agent' and key[2] == 'agent':
                return torch.tensor([[store['agent'].num_nodes], [store['agent'].num_nodes]])
            elif key[0] == 'lane' and key[2] == 'lane':
                return torch.tensor([[store['lane'].num_nodes], [store['lane'].num_nodes]])
            elif key[0] == 'lane' and key[2] == 'agent':
                return torch.tensor([[store['lane'].num_nodes], [store['agent'].num_nodes]])
            elif key[0] == 'agent' and key[2] == 'lane':
                return torch.tensor([[store['agent'].num_nodes], [store['lane'].num_nodes]])
        return super().__inc__(key, value, store)