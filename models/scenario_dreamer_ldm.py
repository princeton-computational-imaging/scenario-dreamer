import os
import pickle
import glob
from tqdm import tqdm
from utils.train_helpers import create_lambda_lr_cosine, create_lambda_lr_linear, create_lambda_lr_constant
from nn_modules.ldm import LDM
from models.scenario_dreamer_autoencoder import ScenarioDreamerAutoEncoder
from utils.data_container import ScenarioDreamerData
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from cfgs.config import PROPORTION_NOCTURNE_COMPATIBLE, NON_PARTITIONED
from utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite
from utils.data_helpers import unnormalize_scene, normalize_latents, unnormalize_latents, convert_batch_to_scenarios, reorder_indices
from utils.inpainting_helpers import normalize_and_crop_scene, sample_num_lanes_agents_inpainting, estimate_heading, sample_route, get_default_route_center_yaw
from utils.torch_helpers import from_numpy
from utils.viz import visualize_batch

import torch 
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch_ema import ExponentialMovingAverage
torch.set_printoptions(sci_mode=False)

class ScenarioDreamerLDM(pl.LightningModule):
    def __init__(self, cfg, cfg_ae):
        super(ScenarioDreamerLDM, self).__init__()

        self.save_hyperparameters()
        self.cfg = cfg 
        self.cfg_model = cfg.model
        self.cfg_dataset = self.cfg.dataset
        self.diff_model = LDM(self.cfg)
        self.autoencoder = ScenarioDreamerAutoEncoder.load_from_checkpoint(self.cfg_model.autoencoder_path, cfg=cfg_ae, map_location='cpu')
        
        self.init_prob_matrix = torch.load(self.cfg.eval.init_prob_matrix_path)
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=self.cfg.train.ema_decay)

    
    def on_train_start(self):
        """ Move ema weights to same device as model"""
        self.ema.to(self.device)


    def optimizer_step(self, *args, **kwargs):
        """ Override the optimizer step to update the EMA weights after each optimizer step."""
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    
    def _log_losses(self, loss_dict, split='train', batch_size=None):
        """ Log the losses to WandB."""
        if split == 'train':
            on_step = True 
            on_epoch = False 
            key_lambda = lambda s: s # no change 
        elif split == 'val':
            on_step = False 
            on_epoch = True
            key_lambda = lambda s: f'val_{s}' # add val_ prefix
        elif split == 'test':
            on_step = False 
            on_epoch = True 
            key_lambda = lambda s: f'test_{s}' # add test_ prefix

        for k,v in loss_dict.items():
            if k == 'loss':
                v = v.item()
            
            self.log(key_lambda(k), v, prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True, batch_size=batch_size)

        if split == 'train':
            cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', cur_lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True) 
    
    
    def training_step(self, data, batch_idx):
        """ Training step for the model. Computes the loss and logs it to WandB."""
        loss_dict = self.diff_model.loss(data)
        self._log_losses(loss_dict, split='train')
        
        return loss_dict['loss']


    def validation_step(self, data, batch_idx):
        """ Validation step for the model. Computes the loss, generates visualizations, and logs it to WandB.
            Using the data object for sample generation allows us to also generate inpainting samples, as the data object contains the ground truth latents.
            before the partition."""
        with self.ema.average_parameters():
            loss_dict = self.diff_model.loss(data)
            self._log_losses(loss_dict, split='val', batch_size=data.batch_size)

            # is_global_zero ensures that only one process logs the visualization
            visualize = self.cfg.train.num_samples_to_visualize > 0 and self.trainer.is_global_zero and batch_idx == 0
            viz_dir = self.cfg.train.viz_dir
            
            num_samples = self.cfg.train.num_samples_to_visualize
            indices = torch.arange(num_samples)
            subset_data_list = data.index_select(indices)
            subset_data = Batch.from_data_list(subset_data_list)  
            
            _, images_to_log = self.forward(
                subset_data,
                'train', # mode
                batch_idx,
                viz_dir,
                visualize=visualize,
                save_wandb=self.cfg.train.track,
                num_samples_to_visualize=self.cfg.train.num_samples_to_visualize
            )
            
            if self.cfg.train.track and visualize:
                self.logger.experiment.log(images_to_log)


    def forward(self, 
                data,
                mode, 
                batch_idx, 
                viz_dir=None, 
                visualize=False, 
                save_wandb=False, 
                num_samples_to_visualize=None):
        """ Forward pass of the model. Generates samples from the diffusion model and decodes them using the autoencoder.
            Also visualizes the generated samples if visualize is True."""
        data = data.to(self.device)
        agent_latents, lane_latents = self.diff_model.forward(data, mode=mode)
        agent_latents, lane_latents = unnormalize_latents(
            agent_latents, 
            lane_latents,
            self.cfg_dataset.agent_latents_mean,
            self.cfg_dataset.agent_latents_std,
            self.cfg_dataset.lane_latents_mean,
            self.cfg_dataset.lane_latents_std
            )
        
        agent_samples, lane_samples, agent_types, lane_types, lane_conn_samples = self.autoencoder.model.forward_decoder(
            agent_latents, 
            lane_latents, 
            data)
        
        agent_samples, lane_samples = unnormalize_scene(
            agent_samples, 
            lane_samples,
            fov=self.cfg_dataset.fov,
            min_speed=self.cfg_dataset.min_speed,
            max_speed=self.cfg_dataset.max_speed,
            min_length=self.cfg_dataset.min_length,
            max_length=self.cfg_dataset.max_length,
            min_width=self.cfg_dataset.min_width,
            max_width=self.cfg_dataset.max_width,
            min_lane_x=self.cfg_dataset.min_lane_x,
            min_lane_y=self.cfg_dataset.min_lane_y,
            max_lane_x=self.cfg_dataset.max_lane_x,
            max_lane_y=self.cfg_dataset.max_lane_y)

        
        if visualize:
            print(f"Visualizing batch {batch_idx}...")
            
            if num_samples_to_visualize is None:
                num_samples_to_visualize = data.batch_size
            
            images_to_log_batch = visualize_batch(
                num_samples_to_visualize, 
                agent_samples, 
                lane_samples, 
                agent_types, 
                lane_types, 
                lane_conn_samples, 
                data, 
                viz_dir, 
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                save_wandb=save_wandb)
        else:
            images_to_log_batch = None

        # no longer latents but now decoded geometric data
        data['agent'].x = agent_samples 
        data['lane'].x = lane_samples 
        data['agent'].type = torch.nn.functional.one_hot(agent_types, num_classes=self.cfg_dataset.num_agent_types)
        if self.cfg.dataset_name == 'nuplan':
            data['lane'].type = torch.nn.functional.one_hot(lane_types, num_classes=self.cfg_dataset.num_lane_types)
        data['lane', 'to', 'lane'].type = lane_conn_samples
        
        return data, images_to_log_batch


    def _build_ldm_dset_from_ae_dset_for_inpainting(self, ae_dset, batch_size, num_samples):
        """ Build a pyg dataset for the LDM from a given autoencoder dataset for inpainting."""
        dataloader = DataLoader(
            ae_dset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        data_list = []
        inpainting_prob_matrix = torch.load(self.cfg.eval.inpainting_prob_matrix_path)
        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)
            agent_latents, lane_latents, lane_cond_dis_prob = self.autoencoder.model.forward_encoder(data)
            
            agent_latents, lane_latents = normalize_latents(
                agent_latents, 
                lane_latents,
                self.cfg_dataset.agent_latents_mean,
                self.cfg_dataset.agent_latents_std,
                self.cfg_dataset.lane_latents_mean,
                self.cfg_dataset.lane_latents_std
            )
            cond_lane_ids = data['lane'].ids
            num_lanes_batch, num_agents_batch = sample_num_lanes_agents_inpainting(
                lane_cond_dis_prob,
                data.map_id,
                data.num_lanes,
                data.num_agents,
                self.cfg_dataset.max_num_lanes,
                inpainting_prob_matrix.to(self.device),
            )

            for i in range(data.batch_size):
                if len(data_list) == num_samples:
                    break 

                d = ScenarioDreamerData()
                num_lanes = num_lanes_batch[i].item()
                num_agents = num_agents_batch[i].item()
                d['num_lanes'] = num_lanes
                d['num_agents'] = num_agents
                d['map_id'] = data['map_id'][i].item()
                d['lg_type'] = data['lg_type'][i].item()

                cond_agent_mask = torch.zeros(num_agents).bool()
                cond_agent_mask[:data['num_agents'][i]] = True
                cond_lane_mask = torch.zeros(num_lanes).bool()
                cond_lane_mask[:data['num_lanes'][i]] = True

                d['agent'].mask = cond_agent_mask 
                d['lane'].mask = cond_lane_mask

                # these two are placeholders
                d['lane'].x = torch.empty((num_lanes, self.cfg_model.lane_latent_dim))
                d['agent'].x = torch.empty((num_agents, self.cfg_model.agent_latent_dim))
                
                agent_latents_i = agent_latents[data['agent'].batch == i]
                agent_states_i = data['agent'].x[data['agent'].batch == i]
                lane_latents_i = lane_latents[data['lane'].batch == i]
                lane_states_i = data['lane'].x[data['lane'].batch == i]
                cond_lane_ids_i = cond_lane_ids[data['lane'].batch == i]

                # we use this function in a hacky way to reorder the latents and conditional lane indices
                # we don't need the updated edge indices here as the lane-to-lane graph is fully connected
                agent_latents_i, _, lane_latents_i, cond_lane_ids_i, _, _, _ = reorder_indices(
                    agent_latents_i.cpu().numpy(), 
                    agent_latents_i.cpu().numpy(), 
                    lane_latents_i.cpu().numpy(), 
                    cond_lane_ids_i.cpu().numpy(), 
                    get_edge_index_complete_graph(len(lane_latents_i)).numpy(), 
                    agent_states_i.cpu().numpy(), 
                    lane_states_i.cpu().numpy(), 
                    lg_type=0, # we don't care about partition masks here
                    dataset=self.cfg.dataset_name)
                agent_latents_i = from_numpy(agent_latents_i)
                lane_latents_i = from_numpy(lane_latents_i)
                cond_lane_ids_i = from_numpy(cond_lane_ids_i)
                
                # conditional latents/lane_ids for the indices corresponding to the lanes/agents after the partition are set to zero
                agents_latents_i_padded = torch.zeros((num_agents, self.cfg_model.agent_latent_dim))
                agents_latents_i_padded[:agent_latents_i.shape[0], :] = agent_latents_i
                agent_latents_i = agents_latents_i_padded
                lane_latents_i_padded = torch.zeros((num_lanes, self.cfg_model.lane_latent_dim))
                lane_latents_i_padded[:lane_latents_i.shape[0], :] = lane_latents_i
                lane_latents_i = lane_latents_i_padded
                cond_lane_ids_i_padded = torch.zeros((num_lanes,))
                cond_lane_ids_i_padded[:cond_lane_ids_i.shape[0]] = cond_lane_ids_i
                cond_lane_ids_i = cond_lane_ids_i_padded

                d['lane'].latents = lane_latents_i
                d['agent'].latents = agent_latents_i
                d['lane'].ids = cond_lane_ids_i

                d['lane', 'to', 'lane'].edge_index = get_edge_index_complete_graph(num_lanes)
                d['agent', 'to', 'agent'].edge_index = get_edge_index_complete_graph(num_agents)
                d['lane', 'to', 'agent'].edge_index = get_edge_index_bipartite(num_lanes, num_agents)

                data_list.append(d)
        
        return data_list


    def _initialize_pyg_dset(self, mode, num_samples, batch_size, conditioning_path=None):
        """ Initialize a PyTorch Geometric dataset with the appropriate metadata for the given generation mode."""
        data_list = []
        map_id_counter = 0
        
        if mode == 'lane_conditioned':
            assert conditioning_path is not None, "conditioning_path must be provided for lane conditioned agent generation"
            # only load non-partitioned scenes
            if self.cfg.dataset_name == 'waymo':
                conditioning_files = sorted(glob.glob(conditioning_path + "/*-of-*_*_0_*.pkl"))
            else:
                conditioning_files = sorted(glob.glob(conditioning_path + "/*_0.pkl"))
            conditioning_files = conditioning_files[:num_samples] 
        
        elif mode == 'inpainting':
            assert conditioning_path is not None, "conditioning_path must be provided for inpainting generation"
            conditioning_files = sorted(glob.glob(conditioning_path + "/*_*.pkl"))
            conditioning_files = conditioning_files[:num_samples]
        
        for i in range(num_samples):
            d = ScenarioDreamerData()

            if mode == 'initial_scene':
                if self.cfg.dataset_name == 'waymo':
                    map_id = torch.multinomial(
                        torch.tensor([1-PROPORTION_NOCTURNE_COMPATIBLE, PROPORTION_NOCTURNE_COMPATIBLE]), 1)
                else:
                    map_id = map_id_counter 
                    map_id_counter += 1
                    map_id_counter = map_id_counter % self.cfg_dataset.num_map_ids

                lane_agent_probs = self.init_prob_matrix[map_id].reshape(1, -1)
                folded_num_lanes_agents = torch.multinomial(lane_agent_probs, 1).squeeze(-1)
                # +1 because there is an index for "no agents" and "no lanes"
                num_lanes = (folded_num_lanes_agents // (self.cfg_dataset.max_num_agents + 1)).item()
                num_agents = (folded_num_lanes_agents % (self.cfg_dataset.max_num_agents + 1)).item()

                assert num_lanes > 0 and num_agents > 0, "Generating scene with either no lanes or no agents"

                lg_type = NON_PARTITIONED # as we are generating initial scenes

                d['map_id'] = int(map_id)
                d['lg_type'] = int(lg_type)
                d['num_lanes'] = int(num_lanes)
                d['num_agents'] = int(num_agents)
                d['lane'].x = torch.empty((num_lanes, self.cfg_model.lane_latent_dim))
                d['agent'].x = torch.empty((num_agents, self.cfg_model.agent_latent_dim))
                d['lane', 'to', 'lane'].edge_index = get_edge_index_complete_graph(num_lanes)
                d['agent', 'to', 'agent'].edge_index = get_edge_index_complete_graph(num_agents)
                d['lane', 'to', 'agent'].edge_index = get_edge_index_bipartite(num_lanes, num_agents)

                data_list.append(d)

            elif mode == 'inpainting':
                conditioning_file = conditioning_files[i]
                with open(os.path.join(conditioning_path, conditioning_file), 'rb') as f:
                    cond_d = pickle.load(f)
                
                if 'route' in cond_d:
                    route = cond_d['route']
                else:
                    route, found_route = sample_route(cond_d, dataset=self.cfg.dataset_name)
                    if found_route:
                        center = route[-1]
                        _, yaw = estimate_heading(route)
                    else:
                        # TODO: Might be better to actually define the route as a straight line
                        center, yaw = get_default_route_center_yaw(dataset=self.cfg.dataset_name)
                        # TODO: indicate that this scene is no longer valid
                        
                # normalize to endpoint of route
                normalize_dict = {
                    'center': center,
                    'yaw': yaw
                }

                d = normalize_and_crop_scene(cond_d, d, normalize_dict, self.cfg_dataset, self.cfg.dataset_name)
                data_list.append(d)

            elif mode == 'lane_conditioned':
                # process conditioning data similar to ldm dataloader
                conditioning_file = conditioning_files[i]
                with open(os.path.join(conditioning_path, conditioning_file), 'rb') as f:
                    cond_d = pickle.load(f)

                agent_states = cond_d['agent_states']
                road_points = cond_d['road_points']
                lane_mu = cond_d['lane_mu']
                agent_mu = cond_d['agent_mu']
                lane_log_var = cond_d['lane_log_var']
                agent_log_var = cond_d['agent_log_var']
                edge_index_lane_to_lane = cond_d['edge_index_lane_to_lane']
                edge_index_lane_to_agent = cond_d['edge_index_lane_to_agent']
                edge_index_agent_to_agent = cond_d['edge_index_agent_to_agent']
                scene_type = cond_d['scene_type']
                if self.cfg.dataset_name == 'nuplan':
                    map_id = cond_d['map_id']
                else:
                    map_id = cond_d['nocturne_compatible']
                num_lanes = lane_mu.shape[0]
                num_agents = agent_mu.shape[0]

                # apply recursive ordering
                agent_mu, agent_log_var, lane_mu, lane_log_var, edge_index_lane_to_lane, _, _ = reorder_indices(
                    agent_mu, 
                    agent_log_var, 
                    lane_mu, 
                    lane_log_var, 
                    edge_index_lane_to_lane, 
                    agent_states, 
                    road_points, 
                    scene_type,
                    dataset=self.cfg.dataset_name)
                edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)
                
                d['map_id'] = map_id
                d['lg_type'] = scene_type
                d['num_lanes'] = num_lanes
                d['num_agents'] = num_agents
                
                _, lane_latents = normalize_latents(
                    torch.empty((num_agents, self.cfg_model.agent_latent_dim)),
                    from_numpy(lane_mu),
                    self.cfg_dataset.agent_latents_mean,
                    self.cfg_dataset.agent_latents_std,
                    self.cfg_dataset.lane_latents_mean,
                    self.cfg_dataset.lane_latents_std
                )
                
                # these two are placeholders
                d['lane'].x = torch.empty((num_lanes, self.cfg_model.lane_latent_dim))
                d['agent'].x = torch.empty((num_agents, self.cfg_model.agent_latent_dim))
                
                # the lane latents will be used in the land-conditioned generation
                d['lane'].latents = lane_latents
                d['lane', 'to', 'lane'].edge_index = from_numpy(edge_index_lane_to_lane)
                d['agent', 'to', 'agent'].edge_index = from_numpy(edge_index_agent_to_agent)
                d['lane', 'to', 'agent'].edge_index = from_numpy(edge_index_lane_to_agent)
                data_list.append(d)
            
        # in inpainting mode, we still need to feed through the autoencoder to get latents
        # and construct the LDM pyg dataset object
        if mode == 'inpainting':
            data_list = self._build_ldm_dset_from_ae_dset_for_inpainting(data_list, batch_size, num_samples)
        
        return data_list


    def generate(
            self,
            mode,
            num_samples,
            batch_size,
            cache_samples=False,
            visualize=False,
            conditioning_path=None,
            cache_dir=None,
            viz_dir=None,
            save_wandb = False,
            return_samples=False,
    ):
        """ Generate samples using the diffusion model."""
        if conditioning_path is not None:
            assert len(os.listdir(conditioning_path)) >= num_samples, f"Not enough conditioning samples in {conditioning_path} to generate {num_samples} samples."
        
        print(f"Generating {num_samples} samples with mode={mode}...")
        
        self.eval()
        with torch.no_grad():
            with self.ema.average_parameters():
                images_to_log = {}
                
                # initialize pyg dataset with appropriate metadata (edge indices, etc.) 
                dset = self._initialize_pyg_dset(
                    mode,
                    num_samples,
                    batch_size,
                    conditioning_path,
                )
                
                dataloader = DataLoader(
                    dset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False
                )
                scenarios = {}
                for batch_idx, data in enumerate(tqdm(dataloader)):
                    # updates data object with generated samples 
                    data, images_to_log_batch = self.forward(
                        data, 
                        mode,
                        batch_idx, 
                        viz_dir=viz_dir, 
                        visualize=visualize, 
                        save_wandb=save_wandb)
                    if visualize and save_wandb:
                        images_to_log.update(images_to_log_batch)
                    batch_of_scenarios = convert_batch_to_scenarios(data, batch_idx=batch_idx, cache_dir=cache_dir, cache_samples=cache_samples, cache_lane_types=self.cfg.dataset_name == 'nuplan')
                    scenarios.update(batch_of_scenarios)
        
        return scenarios if return_samples else None
    
    
    def on_before_optimizer_step(self, optimizer):
        """ Called before the optimizer step. Logs the gradient norms for each layer."""
        # Compute the 2-norm for each layer
        norms_encoder = grad_norm(self.diff_model.model, norm_type=2)
        self.log_dict(norms_encoder)

    
    def on_save_checkpoint(self, checkpoint):
        """ Called when saving a checkpoint. Saves the EMA state dict."""
        checkpoint['ema_state_dict'] = self.ema.state_dict()
    

    def on_load_checkpoint(self, checkpoint):
        """ Called when loading a checkpoint. Loads the EMA state dict."""
        self.ema.load_state_dict(checkpoint['ema_state_dict'])


     ### Taken largely from QCNet repository: https://github.com/ZikangZhou/QCNet
    def configure_optimizers(self):
        """ Configure the optimizer and learning rate scheduler for the model."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.diff_model.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        
        # only optimize the diffusion model
        param_dict = {param_name: param for param_name, param in self.diff_model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.cfg.train.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay, betas=(self.cfg.train.beta_1, self.cfg.train.beta_2), eps=self.cfg.train.epsilon)
        
        if self.cfg.train.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_cosine(self.cfg))
        elif self.cfg.train.lr_schedule == 'linear':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_linear(self.cfg))
        elif self.cfg.train.lr_schedule == 'constant':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_constant(self.cfg))

        return [optimizer], {"scheduler": scheduler,
                             "interval": "step",
                             "frequency": 1}