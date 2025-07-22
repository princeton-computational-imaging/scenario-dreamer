import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union

import torch.nn.functional as F
from utils.layers import ResidualMLP, AttentionLayer, AutoEncoderFactorizedAttentionBlock
from utils.train_helpers import weight_init
from utils.losses import GeometricLosses
from utils.data_container import get_batches, get_features, get_edge_indices, get_encoder_edge_indices
from utils.data_helpers import reparameterize
from cfgs.config import NON_PARTITIONED

class ScenarioDreamerEncoder(nn.Module):
    """Encoder of the Scenario Dreamer AutoEncoder."""
    
    def __init__(self, cfg):
        super(ScenarioDreamerEncoder, self).__init__()
        self.cfg = cfg
        
        # ------------------------------------------------------------------ #
        # Learnable query token used for lane-conditional distribution head  #
        # ------------------------------------------------------------------ #
        self.Q = nn.Parameter(torch.Tensor(1, self.cfg.hidden_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)
        # Lane-conditional distribution head
        self.pred_lane_cond_dis = ResidualMLP(input_dim=self.cfg.hidden_dim, 
                                              hidden_dim=self.cfg.hidden_dim, 
                                              n_hidden=2, 
                                              output_dim=self.cfg.max_num_lanes+1)
        # Fuse lane information into the query token
        self.l2q_transformer_layer = AttentionLayer(hidden_dim=self.cfg.hidden_dim,
                                                    num_heads=self.cfg.num_heads,
                                                    head_dim= self.cfg.hidden_dim // self.cfg.num_heads,
                                                    feedforward_dim = self.cfg.dim_f,
                                                    dropout=self.cfg.dropout,
                                                    bipartite=True,
                                                    has_pos_emb=False,
                                                    pos_emb_hidden_dim=None)
        
        # Embeds vectorized elements and lane connection types
        self.lane_mlp = ResidualMLP(input_dim=self.cfg.num_points_per_lane * self.cfg.lane_attr + self.cfg.num_lane_types, 
                                    hidden_dim=self.cfg.hidden_dim)
        self.agent_mlp = ResidualMLP(input_dim=self.cfg.state_dim + self.cfg.num_agent_types, 
                                     hidden_dim=self.cfg.agent_hidden_dim)
        self.lane_conn_mlp = ResidualMLP(input_dim=self.cfg.lane_conn_attr, 
                                         hidden_dim=self.cfg.lane_conn_hidden_dim)

        # Factorised attention encoder blocks
        self.encoder_transformer_blocks = []
        for l in range(self.cfg.num_encoder_blocks):
            encoder_transformer_block = AutoEncoderFactorizedAttentionBlock(
                lane_hidden_dim=self.cfg.hidden_dim,
                lane_feedforward_dim=self.cfg.dim_f,
                lane_num_heads=self.cfg.num_heads,
                agent_hidden_dim=self.cfg.agent_hidden_dim,
                agent_feedforward_dim=self.cfg.agent_dim_f,
                agent_num_heads=self.cfg.agent_num_heads,
                lane_conn_hidden_dim=self.cfg.lane_conn_hidden_dim,
                dropout=self.cfg.dropout)

            self.encoder_transformer_blocks.append(encoder_transformer_block) 
        self.encoder_transformer_blocks = nn.ModuleList(self.encoder_transformer_blocks)

        # Gaussian latent variable heads
        self.agent_mu = nn.Linear(self.cfg.agent_hidden_dim, self.cfg.agent_latent_dim)
        self.lane_mu = nn.Linear(self.cfg.hidden_dim, self.cfg.lane_latent_dim)
        self.agent_log_var = nn.Linear(self.cfg.agent_hidden_dim, self.cfg.agent_latent_dim)
        self.lane_log_var = nn.Linear(self.cfg.hidden_dim, self.cfg.lane_latent_dim)
        
        self.apply(weight_init)
    
    
    def forward(
        self,
        x_agent: torch.Tensor,
        x_lane: torch.Tensor,
        x_lane_conn: torch.Tensor,
        a2a_edge_index: torch.Tensor,
        l2l_edge_index: torch.Tensor,
        l2a_edge_index: torch.Tensor,
        l2q_edge_index: torch.Tensor,
        agent_batch: torch.Tensor,
        return_lane_embeddings: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        np.ndarray,
    ]:
        """Encode a batch of vectorized scenes (agents + lanes + lane connectivity).

        Args:
            x_agent: Tensor of shape *(N_agents (across batch), 10)*
                containing the per-agent input features.
            x_lane: Tensor of shape *(N_lanes (across batch), num_points_per_lane,
                lane_attr)* of vectorized lane polylines.
            x_lane_conn: Tensor of shape *(N_lane_edges, lane_conn_attr)*
                with categorical lane-to-lane connection attributes.
            a2a_edge_index: COO index *(2, E_agent)* defining agent-to-agent
                edges.
            l2l_edge_index: COO index *(2, E_lane)* defining lane-to-lane
                edges.
            l2a_edge_index: COO index *(2, E_cross)* defining
                lane-to-agent edges.
            l2q_edge_index: COO index *(2, E_query)* connecting **lanes** to
                the learnable **query token**.
            agent_batch: Tensor of shape *(N_agents,)* mapping each agent node
                to its sample in the mini-batch.
            return_lane_embeddings: If ``True`` the method *short-circuits*
                and returns the penultimate layer lane embeddings; otherwise the full set of latent outputs is
                returned.

        Returns:
            *When ``return_lane_embeddings`` is **True**:*

                np.ndarray  
                    An array of shape *(N_lanes, hidden_dim)* on CPU containing
                    the lane embeddings.

            *When ``return_lane_embeddings`` is **False** (default):*

                Tuple[
                    torch.Tensor,  # agent_mu         (N_agents, agent_latent_dim)
                    torch.Tensor,  # lane_mu          (N_lanes,  lane_latent_dim)
                    torch.Tensor,  # agent_log_var    (N_agents, agent_latent_dim)
                    torch.Tensor,  # lane_log_var     (N_lanes,  lane_latent_dim)
                    torch.Tensor   # lane_cond_dis_logits (B, max_num_lanes + 1)
                    torch.Tensor   # lane_cond_dis_prob (B, max_num_lanes + 1)
                ]
        """
        assert x_agent.dtype == torch.float32, "x_agent should be of type float32"
        assert x_lane.dtype == torch.float32, "x_lane should be of type float32"
        assert x_lane_conn.dtype == torch.float32, "x_lane_conn should be of type float32"
        
        agent_embeddings = self.agent_mlp(x_agent)
        lane_embeddings = self.lane_mlp(x_lane)
        lane_conn_embeddings = self.lane_conn_mlp(x_lane_conn)

        batch_size = int((agent_batch.max() + 1).item())
        query_embeddings = self.Q.repeat(batch_size, 1)
        
        for l in range(self.cfg.num_encoder_blocks):
            agent_embeddings, lane_embeddings, lane_conn_embeddings = self.encoder_transformer_blocks[l](agent_embeddings, 
                                                                                                        lane_embeddings, 
                                                                                                        lane_conn_embeddings, 
                                                                                                        lane_conn_embeddings,
                                                                                                        a2a_edge_index,
                                                                                                        l2l_edge_index,
                                                                                                        l2a_edge_index)
            lane_query_embeddings = torch.cat([lane_embeddings, query_embeddings], dim=0)
            lane_query_embeddings = self.l2q_transformer_layer(lane_query_embeddings, None, l2q_edge_index)
            query_embeddings = lane_query_embeddings[lane_embeddings.shape[0]:]
        
        if return_lane_embeddings:
            return lane_embeddings.detach().cpu().numpy()
        
        lane_cond_dis_logits = self.pred_lane_cond_dis(query_embeddings)
        lane_cond_dis_prob = F.softmax(lane_cond_dis_logits, -1)
        agent_mu = self.agent_mu(agent_embeddings)
        lane_mu = self.lane_mu(lane_embeddings)
        agent_log_var = self.agent_log_var(agent_embeddings)
        lane_log_var = self.lane_log_var(lane_embeddings)
        
        return agent_mu, lane_mu, agent_log_var, lane_log_var, lane_cond_dis_logits, lane_cond_dis_prob


class ScenarioDreamerDecoder(nn.Module):
    """Decoder of the Scenario Dreamer AutoEncoder."""
    
    def __init__(self, cfg):
        super(ScenarioDreamerDecoder, self).__init__()
        self.cfg = cfg
        
        # ------------------- linear projections from latent space -------- #
        self.lane_mlp = nn.Linear(self.cfg.lane_latent_dim, self.cfg.hidden_dim)
        self.agent_mlp = nn.Linear(self.cfg.agent_latent_dim, self.cfg.agent_hidden_dim)
        self.downsample_lane_mlp = nn.Linear(self.cfg.hidden_dim, self.cfg.lane_conn_hidden_dim)
        self.lane_conn_mlp = nn.Linear(self.cfg.lane_conn_hidden_dim * 2, self.cfg.lane_conn_hidden_dim)

        # ------------------- factorized attention decoder blocks ---------------------- #
        self.decoder_transformer_blocks = []
        for l in range(self.cfg.num_decoder_blocks):
            decoder_transformer_block = AutoEncoderFactorizedAttentionBlock(
                lane_hidden_dim=self.cfg.hidden_dim,
                lane_feedforward_dim=self.cfg.dim_f,
                lane_num_heads=self.cfg.num_heads,
                agent_hidden_dim=self.cfg.agent_hidden_dim,
                agent_feedforward_dim=self.cfg.agent_dim_f,
                agent_num_heads=self.cfg.agent_num_heads,
                lane_conn_hidden_dim=self.cfg.lane_conn_hidden_dim,
                dropout=self.cfg.dropout)
            self.decoder_transformer_blocks.append(decoder_transformer_block) 
        self.decoder_transformer_blocks = nn.ModuleList(self.decoder_transformer_blocks)

        # ------------------- output heads -------------------------------- #
        self.pred_agent_states = ResidualMLP(input_dim=self.cfg.agent_hidden_dim, 
                                             hidden_dim=self.cfg.agent_hidden_dim, 
                                             n_hidden=3,
                                             output_dim=self.cfg.state_dim)
        self.pred_agent_types = ResidualMLP(input_dim=self.cfg.agent_hidden_dim, 
                                            hidden_dim=self.cfg.agent_hidden_dim, 
                                            n_hidden=2,
                                            output_dim=self.cfg.num_agent_types)
        if self.cfg.num_lane_types > 0:
            self.pred_lane_types = ResidualMLP(input_dim=self.cfg.hidden_dim, 
                                               hidden_dim=self.cfg.hidden_dim, 
                                               n_hidden=2,
                                               output_dim=self.cfg.num_lane_types)
        
        self.pred_lane_states = ResidualMLP(input_dim=self.cfg.hidden_dim, 
                                            hidden_dim=self.cfg.hidden_dim, 
                                            n_hidden=3,
                                            output_dim=self.cfg.num_points_per_lane * self.cfg.lane_attr)
        self.pred_lane_conn = ResidualMLP(input_dim=self.cfg.lane_conn_hidden_dim, 
                                          hidden_dim=self.cfg.lane_conn_hidden_dim, 
                                          n_hidden=2,
                                          output_dim=self.cfg.lane_conn_attr)
        self.apply(weight_init)


    def forward(
        self,
        x_agent: torch.Tensor,
        x_lane: torch.Tensor,
        a2a_edge_index: torch.Tensor,
        l2l_edge_index: torch.Tensor,
        l2a_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent embeddings into vectorized driving scenes.

        Args:
            x_agent: Tensor *(N_agents, agent_latent_dim)* - latent agent
                embeddings sampled from the encoder.
            x_lane: Tensor *(N_lanes, lane_latent_dim)* - latent lane
                embeddings.
            a2a_edge_index: COO index *(2, E_agent)* for agent-to-agent edges.
            l2l_edge_index: COO index *(2, E_lane)* for lane-to-lane edges.
            l2a_edge_index: COO index *(2, E_cross)* for lane-to-agent edges.

        Returns:
            Tuple containing, in order:

            * **agent_states_pred** - *(N_agents, state_dim)* agent state predictions
            * **agent_types_logits** - *(N_agents, num_agent_types)* logits for
              categorical agent type prediction.
            * **agent_types_pred** - *(N_agents,)* predictions for
              categorical agent type prediction.
            * **lane_states_pred** - *(N_lanes, num_points_per_lane,
              lane_attr)* predicted lane vectors
            * **lane_types_logits** - *(N_lanes, num_lane_types) or None* logits for
              categorical lane type prediction.
            * **lane_types_pred** - *(N_lanes,) or None* predictions for
              categorical lane type prediction.
            * **lane_conn_logits** - *(E_lane, lane_conn_attr)* logits for lane
              connectivity classification.
            * **lane_conn_pred** - *(E_lane, 6)* predictions for lane
              connectivity classification as one-hot vectors.
        """
        
        # ----------- latent -> hidden-dim projections -------------------- #
        agent_embeddings = self.agent_mlp(x_agent)
        lane_embeddings = self.lane_mlp(x_lane)

        # ----------- build lane-connection embeddings -------------------- #
        lane_embeddings_downsampled = self.downsample_lane_mlp(lane_embeddings)
        src_lane_conn_embedding = lane_embeddings_downsampled[l2l_edge_index[0]]
        dst_lane_conn_embedding = lane_embeddings_downsampled[l2l_edge_index[1]]
        lane_conn_embeddings = self.lane_conn_mlp(torch.cat([src_lane_conn_embedding, dst_lane_conn_embedding], dim=-1))

        # ----------- factorized attention processing ------------------------ #
        for l in range(self.cfg.num_decoder_blocks):
            agent_embeddings, lane_embeddings, lane_conn_embeddings = self.decoder_transformer_blocks[l](agent_embeddings, 
                                                                                                        lane_embeddings, 
                                                                                                        lane_conn_embeddings,
                                                                                                        lane_conn_embeddings, 
                                                                                                        a2a_edge_index,
                                                                                                        l2l_edge_index,
                                                                                                        l2a_edge_index)

        # ----------- prediction heads ------------------------------------ #
        agent_states_pred = self.pred_agent_states(agent_embeddings)
        lane_states_pred = self.pred_lane_states(lane_embeddings).reshape(x_lane.shape[0], self.cfg.num_points_per_lane, self.cfg.lane_attr)
        
        agent_types_logits = self.pred_agent_types(agent_embeddings)
        agent_types_pred = torch.argmax(agent_types_logits, dim=1)
        
        if self.cfg.num_lane_types > 0:
            lane_types_logits = self.pred_lane_types(lane_embeddings)
            lane_types_pred = torch.argmax(lane_types_logits, dim=1)
        else:
            lane_types_logits = None 
            lane_types_pred = None
        
        lane_conn_logits = self.pred_lane_conn(lane_conn_embeddings)
        lane_conn_pred = torch.argmax(lane_conn_logits, dim=1)
        lane_conn_pred =  F.one_hot(lane_conn_pred, num_classes=self.cfg.lane_conn_attr)
        
        return agent_states_pred, agent_types_logits, agent_types_pred, lane_states_pred, lane_types_logits, lane_types_pred, lane_conn_logits, lane_conn_pred


class AutoEncoder(nn.Module):
    """Scenario Dreamer AutoEncoder."""
    
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        self.cfg = cfg
        self.encoder = ScenarioDreamerEncoder(self.cfg)
        self.decoder = ScenarioDreamerDecoder(self.cfg)
        
        # loss functions for training variational autoencoder
        self.agent_loss_fn = GeometricLosses['l1']()
        self.lane_loss_fn = GeometricLosses['l1']((1,2))
        self.agent_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.lane_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.lane_conn_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.kl_loss_fn = GeometricLosses['kl']()
        
        self.apply(weight_init)
    
    
    def loss(self, data):
        """ Computes the loss for the autoencoder given a batch of data."""

        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
        a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
        a2a_edge_index_encoder, l2l_edge_index_encoder, l2a_edge_index_encoder, l2q_edge_index_encoder, x_lane_conn_encoder = get_encoder_edge_indices(data)
        
        agent_mu, lane_mu, agent_log_var, lane_log_var, lane_cond_dis_logits, lane_cond_dis_prob = self.encoder(
            x_agent, 
            x_lane, 
            x_lane_conn_encoder, 
            a2a_edge_index_encoder, 
            l2l_edge_index_encoder, 
            l2a_edge_index_encoder, 
            l2q_edge_index_encoder, 
            agent_batch)
        
        agent_latents = reparameterize(agent_mu, agent_log_var)
        lane_latents = reparameterize(lane_mu, lane_log_var)
        
        agent_states_pred, agent_types_logits, agent_types_pred, lane_states_pred, lane_types_logits, lane_types_pred, lane_conn_logits, lane_conn_pred = self.decoder(
            agent_latents, 
            lane_latents, 
            a2a_edge_index, 
            l2l_edge_index, 
            l2a_edge_index)

        lg_type = data['lg_type'] # partitioned (1) or non-partitioned (0)
        lane_cond_dis = data['num_lanes_after_origin'] # gt num lanes after partition
        
        ### Loss computation
        # conditional lane distribution predictor loss
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        lane_cond_dis_loss = ce_loss(lane_cond_dis_logits, lane_cond_dis)
        # we only evaluate lane conditional distribution loss on partitioned scenes
        partition_mask = lg_type == NON_PARTITIONED
        assert torch.all(lane_cond_dis[partition_mask] == 0)
        lane_cond_dis_loss[partition_mask] = 0
        # agent vector regression loss
        agent_loss = self.agent_loss_fn(agent_states_pred, x_agent_states, agent_batch)
        # lane vector regression loss
        lane_loss = self.lane_loss_fn(lane_states_pred, x_lane_states, lane_batch)
        
        if self.cfg.num_lane_types > 0:
            lane_type_loss = self.lane_type_loss_fn(lane_types_logits, x_lane_types, lane_batch)
        else:
            lane_type_loss = torch.tensor(0.0, device=agent_types_logits.device, dtype=agent_types_logits.dtype)
        
        # agent type classification loss
        agent_type_loss = self.agent_type_loss_fn(agent_types_logits, x_agent_types, agent_batch)
        # lane connection type classification loss
        lane_conn_loss = self.lane_conn_loss_fn(lane_conn_logits, x_lane_conn, lane_conn_batch) 
        # kl divergence losses
        agent_kl_loss = self.kl_loss_fn(agent_mu, agent_log_var, agent_batch)
        lane_kl_loss = self.kl_loss_fn(lane_mu, lane_log_var, lane_batch)
        kl_loss = agent_kl_loss + lane_kl_loss

        loss = agent_loss + self.cfg.lane_weight * lane_loss + agent_type_loss + lane_type_loss + self.cfg.lane_conn_weight * lane_conn_loss + self.cfg.kl_weight * kl_loss
        lane_conn_loss = lane_conn_loss.mean().detach()
        
        loss = loss + self.cfg.cond_dis_weight * lane_cond_dis_loss
        lane_cond_dis_loss = lane_cond_dis_loss[~partition_mask].mean().detach()
    
        # compute accuracy of lane conditional distribution predictor
        lane_cond_dis_pred_filtered = torch.argmax(lane_cond_dis_prob[~partition_mask], dim=-1)
        lane_cond_dis_acc = (torch.abs(lane_cond_dis_pred_filtered - lane_cond_dis[~partition_mask]) <= 3).float().mean()
        
        loss_dict = {
            'loss': loss.mean(),
            'agent_loss': agent_loss.mean().detach(),
            'lane_loss': lane_loss.mean().detach(),
            'agent_type_loss': agent_type_loss.mean().detach(),
            'lane_type_loss': lane_type_loss.mean().detach(),
            'lane_conn_loss': lane_conn_loss,
            'kl_loss': kl_loss.mean().detach(),
            'lane_cond_dis_loss': lane_cond_dis_loss,
            'lane_cond_dis_acc': lane_cond_dis_acc
        }
        
        return loss_dict
    
    
    def forward_encoder(self, data, return_stats=False, return_lane_embeddings=False):
        """forward pass through the encoder."""
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
        a2a_edge_index_encoder, l2l_edge_index_encoder, l2a_edge_index_encoder, l2q_edge_index_encoder, x_lane_conn_encoder = get_encoder_edge_indices(data)
        
        encoder_output = self.encoder(
            x_agent, 
            x_lane, 
            x_lane_conn_encoder, 
            a2a_edge_index_encoder, 
            l2l_edge_index_encoder, 
            l2a_edge_index_encoder, 
            l2q_edge_index_encoder, 
            agent_batch,
            return_lane_embeddings)
        
        if return_lane_embeddings:
            return encoder_output
        else:
            agent_mu, lane_mu, agent_log_var, lane_log_var, lane_cond_dis_logits, lane_cond_dis_prob = encoder_output

        if return_stats:
            return agent_mu, lane_mu, agent_log_var, lane_log_var
        
        agent_latents = reparameterize(agent_mu, agent_log_var)
        lane_latents = reparameterize(lane_mu, lane_log_var)

        return agent_latents, lane_latents, lane_cond_dis_prob


    def forward_decoder(self, agent_latents, lane_latents, data):
        """forward pass through the decoder."""
        a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
        agent_states_pred, agent_types_logits, agent_types_pred, lane_states_pred, lane_types_logits, lane_types_pred, lane_conn_logits, lane_conn_pred = self.decoder(
            agent_latents, 
            lane_latents, 
            a2a_edge_index, 
            l2l_edge_index, 
            l2a_edge_index)

        return agent_states_pred, lane_states_pred, agent_types_pred, lane_types_pred, lane_conn_pred
    
    
    def forward(self, data, return_latents=False, return_lane_embeddings=False):
        encoder_output = self.forward_encoder(data, return_stats=return_latents, return_lane_embeddings=return_lane_embeddings)

        if return_latents or return_lane_embeddings:
            return encoder_output
        else:
            agent_latents, lane_latents, lane_cond_dis_prob = encoder_output

        agent_states_pred, lane_states_pred, agent_types_pred, lane_types_pred, lane_conn_pred = self.forward_decoder(agent_latents, lane_latents, data)
        
        return agent_states_pred, lane_states_pred, agent_types_pred, lane_types_pred, lane_conn_pred, lane_cond_dis_prob
    



