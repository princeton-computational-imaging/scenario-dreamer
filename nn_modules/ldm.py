import numpy as np
import torch
from torch import nn
from utils.diffusion_helpers import (
    cosine_beta_schedule,
    extract
)
from utils.losses import GeometricLosses
from nn_modules.dit import DiT
from cfgs.config import BEFORE_PARTITION

class LDM(nn.Module):
    def __init__(self, cfg):
        super(LDM, self).__init__()

        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset
        self.model = DiT(cfg)
        
        n_timesteps = self.cfg_model.n_diffusion_timesteps
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.lane_sampling_temperature = self.cfg_model.lane_sampling_temperature

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        loss_type = self.cfg.train.loss_type
        self.lane_loss_fn = GeometricLosses[loss_type]((1,2))
        self.agent_loss_fn = GeometricLosses[loss_type]((1,2))

    
    def predict_start_from_noise(self, x_t, t, noise):
        """ Predict the start of the diffusion chain from the noised sample x_t and noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    
    def q_posterior(self, x_start, x_t, t):
        """ Compute the mean and log variance of the posterior distribution q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    
    def p_mean_variance(self, x_agent, x_lane, data, t_agent, t_lane):
        """ Predict the mean and log variance of the posterior distribution p(x_{t-1} | x_t, x_0)."""
        # noise prediction
        conditional_epsilon_agent, conditional_epsilon_lane = self.model(x_agent, x_lane, data, t_agent, t_lane, unconditional=False)
        unconditional_epsilon_agent, unconditional_epsilon_lane = self.model(x_agent, x_lane, data, t_agent, t_lane, unconditional=True)
        # classifier-free guidance
        epsilon_agent = unconditional_epsilon_agent + self.cfg.train.guidance_scale * (conditional_epsilon_agent - unconditional_epsilon_agent)
        epsilon_lane = unconditional_epsilon_lane + self.cfg.train.guidance_scale * (conditional_epsilon_lane - unconditional_epsilon_lane)
        
        t_agent = t_agent.detach().to(torch.int64)
        t_lane = t_lane.detach().to(torch.int64)

        # given the noise and timestep, predict the start of the diffusion chain
        x_agent_recon = self.predict_start_from_noise(x_agent, t=t_agent, noise=epsilon_agent)
        x_lane_recon = self.predict_start_from_noise(x_lane, t=t_lane, noise=epsilon_lane)

        # mean, log_var of the posterior distribution q(x_t-1 | x_t, x_0)
        model_mean_agent, posterior_log_variance_agent = self.q_posterior(x_start=x_agent_recon, x_t=x_agent, t=t_agent)
        model_mean_lane, posterior_log_variance_lane = self.q_posterior(x_start=x_lane_recon, x_t=x_lane, t=t_lane)
        
        return model_mean_agent, posterior_log_variance_agent, model_mean_lane, posterior_log_variance_lane

    
    @torch.no_grad()
    def p_sample(self, x_agent, x_lane, data, t_agent, t_lane):
        """ Sample from the posterior distribution p(x_{t-1} | x_t, x_0)."""
        b_agent = t_agent.shape[0]
        b_lane = t_lane.shape[0]

        model_mean_agent, model_log_variance_agent, model_mean_lane, model_log_variance_lane = self.p_mean_variance(
            x_agent, 
            x_lane, 
            data, 
            t_agent, 
            t_lane)
        
        noise_agent = torch.randn_like(x_agent)
        noise_lane = torch.randn_like(x_lane)
        
        # no noise when t == 0
        nonzero_mask_agent = (1 - (t_agent == 0).float()).reshape(b_agent, *((1,) * (len(x_agent.shape) - 1)))
        nonzero_mask_lane = (1 - (t_lane == 0).float()).reshape(b_lane, *((1,) * (len(x_lane.shape) - 1)))
        
        # sample from the posterior distribution using reparametrization trick
        next_x_agent = model_mean_agent + nonzero_mask_agent * (model_log_variance_agent).exp().sqrt() * noise_agent
        next_x_lane = model_mean_lane + nonzero_mask_lane * (model_log_variance_lane).exp().sqrt() * noise_lane * self.lane_sampling_temperature
        
        return next_x_agent, next_x_lane

    @torch.no_grad()
    def p_sample_loop(
        self, 
        agent_shape, 
        lane_shape,
        data, 
        device='cuda',
        mode='initial_scene',
        return_diffusion_chain=False):
        """ Generate a batch of samples from the diffusion model."""
        
        agent_batch = data['agent'].batch 
        lane_batch = data['lane'].batch
        batch_size = data.batch_size
        
        x_agent = torch.randn(agent_shape, device=device)
        # conditional generation on existing lane latents
        if mode == 'lane_conditioned':
            x_lane = data['lane'].latents[:, np.newaxis, :].to(device)
        # jointly generate lane and agent latents
        else:
            x_lane = torch.randn(lane_shape, device=device) * self.lane_sampling_temperature

        # for sample visualizations during training, we can condition on the noiseless latents
        # before the partition to visualize inpainting performance.
        if mode == 'train':
            agent_mask = data['agent'].partition_mask == BEFORE_PARTITION
            x_agent[agent_mask] = data['agent'].latents[agent_mask].unsqueeze(1)
            lane_mask = data['lane'].partition_mask == BEFORE_PARTITION
            x_lane[lane_mask] = data['lane'].latents[lane_mask].unsqueeze(1)
        
        if mode == 'inpainting':
            cond_lane_mask = data['lane'].mask
            x_lane[cond_lane_mask] = data['lane'].latents[cond_lane_mask].unsqueeze(1)
            cond_agent_mask = data['agent'].mask
            x_agent[cond_agent_mask] = data['agent'].latents[cond_agent_mask].unsqueeze(1)
        
        # useful for cool visuals :)
        if return_diffusion_chain: diffusion_chain = [(x_agent, x_lane)]

        # simulate reverse diffusion chain
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_agent = timesteps[agent_batch]
            t_lane = timesteps[lane_batch]
            
            x_agent, x_lane = self.p_sample(x_agent, x_lane, data, t_agent, t_lane)

            x_agent = torch.clip(x_agent, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
            if mode == 'lane_conditioned':
                x_lane = data['lane'].latents[:, np.newaxis, :].to(device)
            else:
                # clip outputs to avoid degenerate samples
                x_lane = torch.clip(x_lane, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)

            if mode == 'inpainting':
                cond_lane_mask = data['lane'].mask
                x_lane[cond_lane_mask] = data['lane'].latents[cond_lane_mask].unsqueeze(1)
                cond_agent_mask = data['agent'].mask
                x_agent[cond_agent_mask] = data['agent'].latents[cond_agent_mask].unsqueeze(1)
            
            if mode == 'train':
                agent_mask = data['agent'].partition_mask == BEFORE_PARTITION
                x_agent[agent_mask] = data['agent'].latents[agent_mask].unsqueeze(1)
                lane_mask = data['lane'].partition_mask == BEFORE_PARTITION
                x_lane[lane_mask] = data['lane'].latents[lane_mask].unsqueeze(1)

            if return_diffusion_chain: diffusion_chain.append((x_agent, x_lane))

        if return_diffusion_chain:
            return x_agent[:, 0], x_lane[:, 0], diffusion_chain
        else:
            return x_agent[:, 0], x_lane[:, 0]

    
    @torch.no_grad()
    def forward(self, data, mode='initial_scene'):
        """generate samples from the diffusion model"""
        
        agent_shape = data['agent'].x[:, np.newaxis, :].shape
        lane_shape = data['lane'].x[:, np.newaxis, :].shape

        return self.p_sample_loop(
            agent_shape, 
            lane_shape, 
            data,
            device=data['agent'].x.device,
            mode=mode,
            return_diffusion_chain=False)


    def q_sample(self, x_start, t, noise=None):
        """generate noised sample for training"""
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    
    def p_losses(
            self, 
            x_agent, 
            x_lane, 
            data, 
            t_agent, 
            t_lane):
        """ Compute the loss for the diffusion model."""
        
        # generate noised latents for training
        agent_noise = torch.randn_like(x_agent)
        x_agent_noisy = self.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)
        lane_noise = torch.randn_like(x_lane)
        x_lane_noisy = self.q_sample(x_start=x_lane, t=t_lane, noise=lane_noise)
        
        # for the partitioned scenes, condition on noiseless latents before partition
        agent_mask = data['agent'].partition_mask == BEFORE_PARTITION
        x_agent_noisy[agent_mask] = x_agent[agent_mask]
        lane_mask = data['lane'].partition_mask == BEFORE_PARTITION
        x_lane_noisy[lane_mask] = x_lane[lane_mask]
        
        agent_noise_pred, lane_noise_pred = self.model(x_agent_noisy, x_lane_noisy, data, t_agent, t_lane)

        assert agent_noise.shape == agent_noise_pred.shape
        assert lane_noise.shape == lane_noise_pred.shape

        # if lg_type == PARTITIONED and latent correspond to element BEFORE_PARTITION, no noise is added
        # TODO: probably better to add mask to remove supervision of latents before partition (which by definition get 0 loss)
        agent_mask = data['agent'].partition_mask == BEFORE_PARTITION
        agent_noise[agent_mask] = 0.
        agent_loss = self.agent_loss_fn(agent_noise_pred, agent_noise, data['agent'].batch)
        lane_mask = data['lane'].partition_mask == BEFORE_PARTITION
        lane_noise[lane_mask] = 0.
        lane_batch = data['lane'].batch
        lane_loss = self.lane_loss_fn(lane_noise_pred, lane_noise, lane_batch)
        
        loss = agent_loss + self.cfg.train.lane_weight * lane_loss
        return loss, agent_loss, lane_loss

    
    def loss(self, data):
        """ Sample diffusion timesteps for training and compute the loss for the diffusion model."""
        # batch of agent and lane latents
        x_agent = data['agent'].latents.unsqueeze(1)
        x_lane = data['lane'].latents.unsqueeze(1)
        
        agent_batch = data['agent'].batch
        lane_batch = data['lane'].batch
        batch_size = data.batch_size 

        # batch of random timesteps        
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_agent.device).long()
        t_agent = t[agent_batch]
        t_lane = t[lane_batch]
        
        
        loss, agent_loss, lane_loss = self.p_losses(x_agent, x_lane, data, t_agent, t_lane)
        
        loss_dict = {
            'loss': loss.mean(),
            'agent_loss': agent_loss.mean().detach(),
            'lane_loss': lane_loss.mean().detach()
        }
        
        return loss_dict