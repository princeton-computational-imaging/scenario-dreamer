import torch
import torch.nn as nn

from utils.train_helpers import weight_init, get_causal_mask
from utils.layers import ResidualMLP


class CtRLSimMapEncoder(nn.Module):
    """ Map Encoder for CtRL-Sim model."""
    def __init__(self, cfg):
        super(CtRLSimMapEncoder, self).__init__()
        self.cfg = cfg 
        self.cfg_model = cfg.model 
        self.cfg_dataset = cfg.dataset

        self.map_seeds = nn.Parameter(
            torch.Tensor(1, 1, self.cfg_model.hidden_dim), 
            requires_grad=True
        )
        nn.init.xavier_uniform_(self.map_seeds)
        self.road_pts_encoder = ResidualMLP(
            self.cfg_model.map_attr, 
            self.cfg_model.hidden_dim
        )
        self.road_pts_attn_layer = nn.MultiheadAttention(
            self.cfg_model.hidden_dim, 
            num_heads=self.cfg_model.num_heads, 
            dropout=self.cfg_model.dropout
        )
        self.norm1 = nn.LayerNorm(self.cfg_model.hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.cfg_model.hidden_dim, eps=1e-5)
        self.map_feats = ResidualMLP(
            self.cfg_model.hidden_dim, 
            self.cfg_model.hidden_dim
        )
        self.apply(weight_init)

    def get_road_pts_mask(self, roads):
        """ Get mask for road points and segments."""
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[2]] = False  # Ensures no NaNs due to empty rows.
        return road_segment_mask, road_pts_mask

    def forward(self, data):
        """ Forward pass through the Map Encoder."""
        road_points = data['map'].road_points.float()
        
        batch_size = road_points.shape[0]
        # [batch_size, num_polylines], [batch_size * num_polylines, num_points_per_polyline]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(road_points)
        road_pts_feats = self.road_pts_encoder(
            road_points[:, :, :, :self.cfg_model.map_attr]
        ).view(
            batch_size*self.cfg_dataset.max_num_lanes, 
            self.cfg_dataset.num_points_per_lane, -1
        ).permute(1, 0, 2)
        map_seeds = self.map_seeds.repeat(
            1, 
            batch_size*self.cfg_dataset.max_num_lanes, 
            1
        )
        
        road_seg_emb = self.road_pts_attn_layer(
            query=map_seeds, 
            key=road_pts_feats, 
            value=road_pts_feats,
            key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(
            1, 
            batch_size, 
            self.cfg_dataset.max_num_lanes, 
            -1
        )[0]
        road_segment_mask = ~road_segment_mask 

        return road_seg_emb, road_segment_mask.bool()


class CtRLSimEncoder(nn.Module):
    """ CtRL-Sim Encoder Module."""
    def __init__(self, cfg):
        super(CtRLSimEncoder, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset
        self.action_dim = self.cfg_dataset.vocab_size

        self.map_encoder = CtRLSimMapEncoder(self.cfg)
        self.embed_state = ResidualMLP(
            self.cfg_model.state_dim, 
            self.cfg_model.hidden_dim
        )
        self.embed_action = nn.Embedding(
            int(self.action_dim), 
            self.cfg_model.hidden_dim
        )
        self.embed_rtg_veh = nn.Embedding(
            self.cfg_dataset.rtg_discretization, 
            self.cfg_model.hidden_dim
        )
        self.embed_rtg = nn.Linear(
            self.cfg_model.hidden_dim * self.cfg_model.num_reward_components, 
            self.cfg_model.hidden_dim
        )
        self.embed_timestep = nn.Embedding(
            self.cfg_dataset.train_context_length, 
            self.cfg_model.hidden_dim
        )
        self.embed_agent_id = nn.Embedding(
            self.cfg_dataset.max_num_agents, 
            self.cfg_model.hidden_dim
        )
        self.embed_ln = nn.LayerNorm(self.cfg_model.hidden_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.cfg_model.hidden_dim, 
                nhead=self.cfg_model.num_heads,
                dim_feedforward=self.cfg_model.dim_feedforward,
                batch_first=True), 
            num_layers=self.cfg_model.num_transformer_encoder_layers)
        self.apply(weight_init)


    def forward(self, data, eval):
        """ Forward pass through the CtRL-Sim Encoder."""
        agent_states = data['agent'].agent_states
        batch_size = agent_states.shape[0]
        seq_len = agent_states.shape[2]
        existence_mask = agent_states[:, :, :, -1:]
        rtg_mask = data['agent'].rtg_mask * existence_mask
        agent_types = data['agent'].agent_types
            
        actions = data['agent'].actions
        agent_ids = torch.arange(
            self.cfg_dataset.max_num_agents).to(agent_states.device)
        # [batch_size, n_agents, timesteps]
        agent_ids = agent_ids.unsqueeze(0).unsqueeze(2).repeat(
            batch_size, 1, agent_states.shape[2])
        rtgs = data['agent'].rtgs
        timesteps = data['agent'].timesteps

        # [batch_size, timesteps, n_agents, 5]
        agent_types = agent_types.unsqueeze(2).repeat(1, 1, agent_states.shape[2], 1).transpose(1,2)

        # [batch_size, timesteps, n_agents, num_actions]
        actions = actions.transpose(1,2)
        # [batch_size, timesteps, n_agents, num_reward_components]
        rtgs = rtgs.transpose(1,2)
        # [batch_size, timesteps, n_agents, 7]
        agent_states = agent_states[:, :, :, :-1].transpose(1,2)
        # [batch_size, timesteps, n_agents, 1]
        timesteps = timesteps.transpose(1,2)
        agent_ids = agent_ids.transpose(1,2)
        # [batch_size, timesteps, n_agents, 1]
        existence_mask = existence_mask.transpose(1,2)
        rtg_mask = rtg_mask.transpose(1,2)
        states = torch.cat([agent_states, agent_types], dim=-1)
        
        if self.cfg_model.encode_initial_state:
            initial_existence_mask = existence_mask[:, 0]

        existence_mask = existence_mask.reshape(batch_size,seq_len*self.cfg_dataset.max_num_agents, 1)
        rtg_mask = rtg_mask.reshape(batch_size,seq_len*self.cfg_dataset.max_num_agents, 1)
        timesteps =  timesteps.reshape(batch_size,seq_len*self.cfg_dataset.max_num_agents)
        agent_ids = agent_ids.reshape(batch_size, seq_len*self.cfg_dataset.max_num_agents)
        states = states.reshape(batch_size, seq_len*self.cfg_dataset.max_num_agents, self.cfg_model.state_dim).float() 
        actions = actions.reshape(batch_size, seq_len*self.cfg_dataset.max_num_agents)
        rtgs = rtgs.reshape(batch_size, seq_len*self.cfg_dataset.max_num_agents, self.cfg_model.num_reward_components)

        timestep_embeddings = self.embed_timestep(timesteps)
        agent_id_embeddings = self.embed_agent_id(agent_ids)
        state_embeddings = self.embed_state(states)
        
        # no goal conditioning
        state_embeddings = state_embeddings + timestep_embeddings + agent_id_embeddings
        
        if self.cfg_model.encode_initial_state:
            initial_state_embeddings = state_embeddings[:, 0:self.cfg_dataset.max_num_agents]
        
        action_embeddings = self.embed_action(actions.long()) + timestep_embeddings + agent_id_embeddings
        
        rtg_veh_embeddings = self.embed_rtg_veh(rtgs[:, :, 0].long())
        rtg_embeddings = self.embed_rtg(rtg_veh_embeddings) + timestep_embeddings + agent_id_embeddings

        # zero out embeddings for missing timesteps
        state_embeddings = state_embeddings * existence_mask.float()
        action_embeddings = action_embeddings * existence_mask.float()
        rtg_embeddings = rtg_embeddings * rtg_mask.float()
        
        if self.cfg_model.encode_initial_state:
            initial_state_embeddings = initial_state_embeddings * initial_existence_mask.float()
            initial_existence_mask = initial_existence_mask[:, :, 0].bool()

        if self.cfg_model.trajeglish:
            stacked_embeddings = (
                action_embeddings.unsqueeze(1).permute(0, 2, 1, 3).reshape(
                    batch_size, 
                    1*seq_len*self.cfg_dataset.max_num_agents, 
                    self.cfg_model.hidden_dim)
            )
        elif self.cfg_model.il:
            stacked_embeddings = torch.stack(
                (state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(
                batch_size, 
                seq_len*self.cfg_dataset.max_num_agents*2, 
                self.cfg_model.hidden_dim
            )
        else:
            stacked_embeddings = torch.stack(
                (state_embeddings, 
                 rtg_embeddings, 
                 action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(
                batch_size, 
                seq_len*self.cfg_dataset.max_num_agents*3, 
                self.cfg_model.hidden_dim
            )
        stacked_embeddings = self.embed_ln(stacked_embeddings)

        polyline_embeddings, valid_mask = self.map_encoder(data)
        
        if self.cfg_model.encode_initial_state:
            pre_encoder_embeddings = torch.cat([polyline_embeddings, initial_state_embeddings], dim=1)
            # we use ~ as "True" corresponds to "ignored" in src_key_padding_mask
            src_key_padding_mask = ~torch.cat([valid_mask, initial_existence_mask], dim=1)
        else:
            pre_encoder_embeddings = polyline_embeddings
            # we use ~ as "True" corresponds to "ignored" in src_key_padding_mask
            src_key_padding_mask = ~valid_mask

        encoder_embeddings = self.transformer_encoder(
            pre_encoder_embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        return {
            'stacked_embeddings': stacked_embeddings, 
            'encoder_embeddings': encoder_embeddings, 
            'src_key_padding_mask': src_key_padding_mask
        }


class CtRLSimDecoder(nn.Module):
    """ Decoder module for CtRL-Sim model."""
    def __init__(self, cfg):
        super(CtRLSimDecoder, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset
        self.action_dim = self.cfg_dataset.vocab_size
        
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.cfg_model.hidden_dim, 
                dim_feedforward=self.cfg_model.dim_feedforward,
                nhead=self.cfg_model.num_heads,
                batch_first=True), 
                num_layers=self.cfg_model.num_decoder_layers)
        
        self.predict_action = ResidualMLP(
                input_dim=self.cfg_model.hidden_dim, 
                hidden_dim=self.cfg_model.hidden_dim, 
                n_hidden=2,
                output_dim=self.action_dim
            )

        if self.cfg_model.predict_rtg:
            self.predict_rtg = ResidualMLP(
                input_dim=self.cfg_model.hidden_dim, 
                hidden_dim=self.cfg_model.hidden_dim, 
                n_hidden=2,
                output_dim=self.cfg_dataset.rtg_discretization * self.cfg_model.num_reward_components
            )

        if not (self.cfg_model.trajeglish or self.cfg_model.il):
            num_types = 3
        elif self.cfg_model.trajeglish:
            num_types = 1
        else:
            num_types = 2
        self.causal_mask = get_causal_mask(self.cfg, self.cfg_dataset.train_context_length, num_types)
        self.apply(weight_init)


    def forward(self, data, scene_enc, eval=False):
        """ Forward pass through the CtRL-Sim decoder."""
        agent_states = data['agent'].agent_states
        batch_size = agent_states.shape[0]
        seq_len = agent_states.shape[2]
        
        # [batch_size, num_timesteps * num_agents * 3, hidden_dim]
        stacked_embeddings = scene_enc['stacked_embeddings']
        # [batch_size, num_polyline_tokens + num_initial_state_tokens, hidden_dim]
        encoder_embeddings = scene_enc['encoder_embeddings']
        # [batch_size, num_polyline_tokens + num_initial_state_tokens]
        src_key_padding_mask = scene_enc['src_key_padding_mask']
        
        output = self.transformer_decoder(
            stacked_embeddings, 
            encoder_embeddings, 
            tgt_mask=self.causal_mask.to(stacked_embeddings.device), 
            memory_key_padding_mask=src_key_padding_mask
        )
        
        preds = {}
        if not (self.cfg_model.trajeglish or self.cfg_model.il):
            # [batch_size, 3, num_timesteps * num_agents, hidden_dim]
            output = output.reshape(
                batch_size, 
                seq_len*self.cfg_dataset.max_num_agents, 
                3, 
                self.cfg_model.hidden_dim
            ).permute(0, 2, 1, 3)
            action_preds = self.predict_action(output[:, 1])
        elif self.cfg_model.trajeglish:
            output = output.reshape(
                batch_size, 
                seq_len*self.cfg_dataset.max_num_agents, 
                1, 
                self.cfg_model.hidden_dim
            ).permute(0, 2, 1, 3)
            action_preds = self.predict_action(output[:, 0])
        else:
            output = output.reshape(
                batch_size, 
                seq_len*self.cfg_dataset.max_num_agents, 
                2, 
                self.cfg_model.hidden_dim
            ).permute(0, 2, 1, 3)
            action_preds = self.predict_action(output[:, 0])
        # [batch_size, num_agents, num_timesteps, action_dim]
        action_preds = action_preds.reshape(
            batch_size, 
            seq_len, 
            self.cfg_dataset.max_num_agents, 
            self.action_dim
        ).permute(0, 2, 1, 3)
        preds['action_preds'] = action_preds
        
        if self.cfg_model.predict_rtg:
            rtg_preds = self.predict_rtg(output[:, 0])
            rtg_preds = rtg_preds.reshape(
                batch_size, 
                seq_len, 
                self.cfg_dataset.max_num_agents, 
                self.cfg_dataset.rtg_discretization * self.cfg_model.num_reward_components
            ).permute(0, 2, 1, 3)
            preds['rtg_preds'] = rtg_preds

        return preds