import torch

def get_edge_index_bipartite(num_src, num_dst):
    """Create a fully connected bipartite `edge_index` tensor from `num_src` to `num_dst` nodes."""
    # Create a meshgrid of all possible combinations of source and destination nodes
    src_indices = torch.arange(num_src)
    dst_indices = torch.arange(num_dst)
    src, dst = torch.meshgrid(src_indices, dst_indices, indexing='ij')

    # Flatten the meshgrid and stack them to create the edge_index
    edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
    return edge_index


def get_edge_index_complete_graph(graph_size):
    """Create a directed complete-graph `edge_index` tensor of size `graph_size`."""
    edge_index = torch.cartesian_prod(torch.arange(graph_size, dtype=torch.long),
                                      torch.arange(graph_size, dtype=torch.long)).t()

    return edge_index


def get_indices_within_scene(batch):
    """Get indices of nodes for each scene in the batch, where for each batch element the indices start from 0."""
    _, counts = batch.unique(return_counts=True)
    index = torch.arange(batch.size(0), device=batch.device) - torch.cumsum(counts, dim=0).repeat_interleave(counts) + counts.repeat_interleave(counts)
    return index