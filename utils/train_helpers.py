import torch.nn as nn
import torch
import math
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle
from datasets.waymo.dataset_ldm_waymo import WaymoDatasetLDM
from datasets.nuplan.dataset_ldm_nuplan import NuplanDatasetLDM
from utils.data_helpers import sample_latents

def create_lambda_lr_cosine(cfg):
    """cosine learning rate schedule with warmup"""
    return lambda current_step: (
        current_step / cfg.train['warmup_steps'] if current_step < cfg.train['warmup_steps']
        else max(
            0.0,
            0.5 * (1 + math.cos(math.pi * (current_step - cfg.train['warmup_steps']) / (cfg.train['max_steps'] - cfg.train['warmup_steps'])))
        )
    )


def create_lambda_lr_linear(cfg):
    """linear learning rate schedule with warmup"""
    return lambda current_step: (
        current_step / cfg.train['warmup_steps'] if current_step < cfg.train['warmup_steps']
        else max(
            0.0,
            (cfg.train['max_steps'] - current_step) / (cfg.train['max_steps'] - cfg.train['warmup_steps'])
        )
    )


def create_lambda_lr_constant(cfg):
    """constant learning rate schedule with warmup"""
    return lambda current_step: (
        current_step / cfg.train['warmup_steps'] if current_step < cfg.train['warmup_steps']
        else 1
    )


def weight_init(m):
    """Initialize weights of PyTorch modules. Inspired by QCNET: https://github.com/ZikangZhou/QCNet"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def cache_latent_stats(cfg):
    """Cache the latent statistics (mean/std) for agent and lane latents computed from the training dataset."""
    if cfg.dataset_name == 'waymo':
        dset = WaymoDatasetLDM(cfg.dataset, split_name='train')
    else:
        dset = NuplanDatasetLDM(cfg.dataset, split_name='train')
    print("Caching latent stats (mean/std of agent/lane latents). Size of dataset: ", len(dset))
    
    os.makedirs(cfg.dataset.latent_stats_dir, exist_ok=True)
    # placeholders to be filled later
    cfg.dataset.agent_latents_mean = 0.0
    cfg.dataset.agent_latents_std = 1.0
    cfg.dataset.lane_latents_mean = 0.0
    cfg.dataset.lane_latents_std = 1.0
    
    dloader = DataLoader(dset, 
               batch_size=cfg.train.batch_size_compute_stats, 
               shuffle=True, 
               num_workers=cfg.datamodule.num_workers,
               pin_memory=True,
               drop_last=True)

    agent_latents_all = []
    lane_latents_all = []
    for i, d in enumerate(tqdm(dloader)):
        if i == cfg.train.num_batches_compute_stats:
            break
        
        agent_latents, lane_latents = sample_latents(
            d, 
            cfg.dataset.agent_latents_mean,
            cfg.dataset.agent_latents_std,
            cfg.dataset.lane_latents_mean,
            cfg.dataset.lane_latents_std,
            normalize=False)
        agent_latents_all.append(agent_latents)
        lane_latents_all.append(lane_latents)
    
    agent_latents_all = torch.cat(agent_latents_all, dim=0)
    lane_latents_all = torch.cat(lane_latents_all, dim=0)

    agent_mean = agent_latents_all.mean().item()
    agent_std = agent_latents_all.std().item()
    lane_mean = lane_latents_all.mean().item()
    lane_std = lane_latents_all.std().item()

    # Pretty print
    print(f"Agent Latents Mean: {agent_mean:.4f}, Std: {agent_std:.4f}")
    print(f"Lane Latents Mean: {lane_mean:.4f}, Std: {lane_std:.4f}")

    # Save to pickle
    stats = {
        'agent_latents_mean': agent_mean,
        'agent_latents_std': agent_std,
        'lane_latents_mean': lane_mean,
        'lane_latents_std': lane_std,
    }
    with open(cfg.dataset.latent_stats_path, 'wb') as f:
        pickle.dump(stats, f)


def set_latent_stats(cfg):
    """Set the latent statistics (mean/std) for agent and lane latents from a cached file."""
    # Load the latent stats from the cached file
    with open(cfg.dataset.latent_stats_path, 'rb') as f:
        stats = pickle.load(f)

    cfg.dataset.agent_latents_mean = stats['agent_latents_mean']
    cfg.dataset.agent_latents_std = stats['agent_latents_std']
    cfg.dataset.lane_latents_mean = stats['lane_latents_mean']
    cfg.dataset.lane_latents_std = stats['lane_latents_std']

    print("Set latent stats from cached file:")
    print(f"Agent Latents Mean: {cfg.dataset.agent_latents_mean:.4f}, Std: {cfg.dataset.agent_latents_std:.4f}")
    print(f"Lane Latents Mean: {cfg.dataset.lane_latents_mean:.4f}, Std: {cfg.dataset.lane_latents_std:.4f}")

    return cfg