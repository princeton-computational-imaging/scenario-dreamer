import numpy as np
import torch
np.set_printoptions(suppress=True)
from torch_geometric.data import HeteroData


class CtRLSimData(HeteroData):
    """ Torch-Geometric `HeteroData` that auto-increments batch indices for proper batching."""
    def __inc__(self, key, value, store):
        return 0


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


def get_batches(data):
    """Extracts agent, lane, and lane connection batch indices from the data object."""
    agent_batch = data['agent'].batch
    lane_batch = data['lane'].batch
    lane_row = data['lane', 'to', 'lane'].edge_index[0]
    lane_conn_batch = lane_batch[lane_row]

    return agent_batch, lane_batch, lane_conn_batch


def get_features(data):
    """Extracts lane and agent features from the data object."""
    x_agent_states = data['agent'].x.float()
    x_agent_types = data['agent'].type.float()
    x_agent = torch.cat([x_agent_states, x_agent_types], dim=-1)
    x_lane_states = data['lane'].x.float()

    if 'type' in data['lane']:
        x_lane_types = data['lane'].type.float()
        x_lane = torch.cat([x_lane_states.reshape(-1, x_lane_states.shape[-2] * x_lane_states.shape[-1]), x_lane_types], dim=-1)
    else:
        x_lane_types = None
        x_lane = x_lane_states.reshape(x_lane_states.shape[0], -1)
    
    x_lane_conn = data['lane', 'to', 'lane'].type.float()

    assert x_agent.dtype == torch.float32, "x_agent should be of type float32"
    assert x_agent_states.dtype == torch.float32, "x_agent_states should be of type float32"
    assert x_agent_types.dtype == torch.float32, "x_agent_types should be of type float32"
    assert x_lane_states.dtype == torch.float32, "x_lane_states should be of type float32"
    assert x_lane.dtype == torch.float32, "x_lane should be of type float32"
    assert x_lane_conn.dtype == torch.float32, "x_lane_conn should be of type float32"

    return x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn

    
def get_edge_indices(data):
    """Extracts edge indices for agent-to-agent, lane-to-lane, and lane-to-agent connections from the data object."""
    a2a_edge_index = data['agent', 'to', 'agent'].edge_index
    l2l_edge_index = data['lane', 'to', 'lane'].edge_index
    l2a_edge_index = data['lane', 'to', 'agent'].edge_index.clone()
    # agent indices are shifted by the number of lanes (as agents are appended to the end of the lane nodes)
    l2a_edge_index[1] = l2a_edge_index[1] + data['lane'].x.shape[0]
    
    assert a2a_edge_index.dtype == torch.int64, "a2a_edge_index should be of type int64"
    assert l2l_edge_index.dtype == torch.int64, "l2l_edge_index should be of type int64"
    assert l2a_edge_index.dtype == torch.int64, "l2a_edge_index should be of type int64"

    return a2a_edge_index, l2l_edge_index, l2a_edge_index


def get_encoder_edge_indices(data):
    """Applies masking to obtain edge indices for agent-to-agent, lane-to-lane, and lane-to-agent connections for the autoencoder encoder."""
    
    # masking for encoder (only applied in partitioned mode as lanes/agents can't attend to other lanes/agents outside their partition)
    a2a_mask = data['agent', 'to', 'agent'].encoder_mask 
    l2l_mask = data['lane', 'to', 'lane'].encoder_mask 
    l2a_mask = data['lane', 'to', 'agent'].encoder_mask
    
    a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
    x_lane_conn = data['lane', 'to', 'lane'].type
    
    a2a_edge_index_encoder = a2a_edge_index[:, a2a_mask]
    l2l_edge_index_encoder = l2l_edge_index[:, l2l_mask]
    l2a_edge_index_encoder = l2a_edge_index[:, l2a_mask]
    x_lane_conn_encoder = x_lane_conn[l2l_mask].float()

    lane_batch = data['lane'].batch
    x_lane = data['lane'].x.float()
    lane_partition_mask = data['lane'].partition_mask
    src = torch.arange(lane_batch.shape[0], device=lane_batch.device)[lane_partition_mask].unsqueeze(0)
    dst = lane_batch[lane_partition_mask].unsqueeze(0) + x_lane.shape[0]
    l2q_edge_index_encoder = torch.cat([src, dst], dim=0)

    assert x_lane_conn_encoder.dtype == torch.float32, "x_lane_conn_encoder should be of type float32"
    
    return a2a_edge_index_encoder, l2l_edge_index_encoder, l2a_edge_index_encoder, l2q_edge_index_encoder, x_lane_conn_encoder