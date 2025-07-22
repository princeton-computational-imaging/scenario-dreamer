import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from utils.train_helpers import weight_init

class ResidualMLP(nn.Module):
    """
    Residual feed-forward block with a configurable number of hidden layers

    Args
    ----
    input_dim   : int - size of the input features
    hidden_dim  : int - size of the hidden / residual features
    n_hidden    : int - number of (Linear → LayerNorm) pairs *before* the residual
    output_dim  : Optional[int] - if given, an extra Linear(hidden_dim, output_dim)
                   is appended *after* the ReLU that follows the residual addition.
                   If None, the block returns the hidden representation directly.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_hidden: int = 2, output_dim: int | None = None):
        super().__init__()

        assert n_hidden >= 1, "Need at least one hidden layer"

        # Main path
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(n_hidden)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_hidden)])

        # Residual (“transform”) path
        self.transform_linear = nn.Linear(input_dim, hidden_dim)
        self.transform_norm = nn.LayerNorm(hidden_dim)

        # Optional final projection
        self.linear_out = nn.Linear(hidden_dim, output_dim) if output_dim is not None else None

        self.relu = nn.ReLU(inplace=True)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, (lin, norm) in enumerate(zip(self.linears, self.norms)):
            out = lin(out)
            out = norm(out)
            # Original blocks *skip* the ReLU on the **last** hidden layer
            if i < len(self.linears) - 1:
                out = self.relu(out)

        # Residual connection
        res = self.transform_norm(self.transform_linear(x))
        out = self.relu(out + res)

        if self.linear_out is not None:
            out = self.linear_out(out)
        return out
    

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) with a single hidden layer."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(x)


class AttentionLayer(MessagePassing):
    """Transformer attention layer, inspired by https://github.com/ZikangZhou/QCNet/blob/main/layers/attention_layer.py."""
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 head_dim,
                 feedforward_dim,
                 dropout,
                 bipartite,
                 has_pos_emb,
                 pos_emb_hidden_dim,
                 **kwargs):
        super(AttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feedforward_dim = feedforward_dim
        self.has_pos_emb = has_pos_emb
        self.pos_emb_hidden_dim = pos_emb_hidden_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(pos_emb_hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(pos_emb_hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, self.feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.feedforward_dim, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(pos_emb_hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self, x, r, edge_index):
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    def message(self, q_i, k_j, v_j, r, index, ptr):
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self, inputs, x_dst):
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(self, x_src, x_dst, r, edge_index):
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        return self.to_out(agg)

    def _ff_block(self, x):
        return self.ff_mlp(x)


class EdgeFeatureUpdate(MessagePassing):
    """Update edge features based on node features and existing edge features."""
    def __init__(self, node_hidden_dim, edge_hidden_dim):
        super(EdgeFeatureUpdate, self).__init__(aggr='add')
        
        self.node_mlp = ResidualMLP(input_dim=node_hidden_dim * 2, 
                                    hidden_dim=edge_hidden_dim)
        self.edge_mlp = ResidualMLP(input_dim=edge_hidden_dim * 2, 
                                    hidden_dim=edge_hidden_dim)
        self.apply(weight_init)

    def forward(self, x, edge_index, edge_attr):
        return self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

    def edge_update(self, x_i, x_j, edge_attr):
        node_features = torch.cat([x_i, x_j], dim=-1)
        updated_node_features = self.node_mlp(node_features)
        combined_features = torch.cat([updated_node_features, edge_attr], dim=-1)
        new_edge_attr = self.edge_mlp(combined_features)

        return new_edge_attr
    

class AutoEncoderFactorizedAttentionBlock(nn.Module):
    """Factorized Transformer block for autoencoder architecture"""
    def __init__(self, 
                 lane_hidden_dim,
                 lane_feedforward_dim,
                 lane_num_heads,
                 agent_hidden_dim,
                 agent_feedforward_dim,
                 agent_num_heads,
                 lane_conn_hidden_dim,
                 dropout):
        
        super(AutoEncoderFactorizedAttentionBlock, self).__init__()
        self.lane_hidden_dim = lane_hidden_dim 
        self.lane_feedforward_dim = lane_feedforward_dim 
        self.lane_num_heads = lane_num_heads 
        self.agent_hidden_dim = agent_hidden_dim 
        self.agent_feedforward_dim = agent_feedforward_dim 
        self.agent_num_heads = agent_num_heads 
        self.lane_conn_hidden_dim = lane_conn_hidden_dim 
        self.dropout = dropout

        self.a2a_transformer_layer = AttentionLayer(hidden_dim=self.agent_hidden_dim,
                                                    num_heads=self.agent_num_heads,
                                                    head_dim= self.agent_hidden_dim // self.agent_num_heads,
                                                    feedforward_dim = self.agent_feedforward_dim,
                                                    dropout=self.dropout,
                                                    bipartite=False,
                                                    has_pos_emb=False,
                                                    pos_emb_hidden_dim=None)

        self.l2l_transformer_layer = AttentionLayer(hidden_dim=self.lane_hidden_dim,
                                                    num_heads=self.lane_num_heads,
                                                    head_dim= self.lane_hidden_dim // self.lane_num_heads,
                                                    feedforward_dim = self.lane_feedforward_dim,
                                                    dropout=self.dropout,
                                                    bipartite=False,
                                                    has_pos_emb=True,
                                                    pos_emb_hidden_dim=self.lane_conn_hidden_dim)

        self.l2a_transformer_layer = AttentionLayer(hidden_dim=self.agent_hidden_dim,
                                                    num_heads=self.agent_num_heads,
                                                    head_dim= self.agent_hidden_dim // self.agent_num_heads,
                                                    feedforward_dim = self.agent_feedforward_dim,
                                                    dropout=self.dropout,
                                                    bipartite=True,
                                                    has_pos_emb=False,
                                                    pos_emb_hidden_dim=None)


        self.update_edge_embeddings = EdgeFeatureUpdate(node_hidden_dim=self.agent_hidden_dim, # downsampled lane hidden_dim
                                                        edge_hidden_dim=self.lane_conn_hidden_dim)
        self.downsample_lane_emb = nn.Linear(self.lane_hidden_dim, self.agent_hidden_dim)
        self.apply(weight_init)

    
    def forward(self, agent_embeddings, 
                      lane_embeddings, 
                      lane_conn_embeddings,
                      edge_emb, 
                      a2a_edge_index,
                      l2l_edge_index,
                      l2a_edge_index):
        
        lane_embeddings = self.l2l_transformer_layer(lane_embeddings, lane_conn_embeddings, l2l_edge_index)
        lane_embeddings_downsampled = self.downsample_lane_emb(lane_embeddings)
        agent_dim_embeddings = torch.cat([lane_embeddings_downsampled, agent_embeddings], dim=0)
        agent_dim_embeddings = self.l2a_transformer_layer(agent_dim_embeddings, None, l2a_edge_index)
        agent_embeddings = agent_dim_embeddings[len(lane_embeddings):]
        agent_embeddings = self.a2a_transformer_layer(agent_embeddings, None, a2a_edge_index)

        lane_conn_embeddings = self.update_edge_embeddings(lane_embeddings_downsampled, l2l_edge_index, lane_conn_embeddings)
        return agent_embeddings, lane_embeddings, lane_conn_embeddings
