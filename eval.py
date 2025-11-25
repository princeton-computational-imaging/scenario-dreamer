import os 
import hydra
from omegaconf import OmegaConf
from models.scenario_dreamer_autoencoder import ScenarioDreamerAutoEncoder
from models.scenario_dreamer_ldm import ScenarioDreamerLDM
from metrics import Metrics

import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.strategies import DDPStrategy
from cfgs.config import CONFIG_PATH
from utils.train_helpers import set_latent_stats
import utils.sim_env_helpers as _sim_env_helpers


def generate_simulation_environments(cfg, cfg_ae, save_dir=None):
    """ Generate simulation environments using the Scenario Dreamer Latent Diffusion Model.
    
    This involves 1 step of initial scene generation followed by multiple steps of
    inpainting to extend the scenario until the desired route length is reached.
    Additional rule-based heuristics are applied to ensure scenario validity.
    """
    cfg = set_latent_stats(cfg)

    # load last ckpt for inference
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            print("Loading checkpoint: ", ckpt_path)
            break
    
    assert ckpt_path is not None, "No checkpoint found in the save directory."

    model = ScenarioDreamerLDM.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae).to('cuda')
    _sim_env_helpers.generate_simulation_environments(model, cfg, save_dir)


def eval_ldm(cfg, cfg_ae, save_dir=None):
    """ Evaluate the Scenario Dreamer Latent Diffusion Model."""
    if cfg.eval.mode == 'metrics':
        metric_evaluator = Metrics(cfg, cfg_ae)
        metric_evaluator.compute_metrics()
        return
    
    cfg = set_latent_stats(cfg)
    
    # load last ckpt for inference
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            print("Loading checkpoint: ", ckpt_path)
            break
    
    assert ckpt_path is not None, "No checkpoint found in the save directory."
    
    # generate samples
    model = ScenarioDreamerLDM.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae).to('cuda')
    model.generate(
        mode = cfg.eval.mode, # Scenario Dreamer supports multiple generation modes: initial_scene, lane_conditioned, and inpainting
        num_samples = cfg.eval.num_samples,
        batch_size = cfg.eval.batch_size,
        cache_samples = cfg.eval.cache_samples,
        visualize = cfg.eval.visualize,
        conditioning_path = cfg.eval.conditioning_path,
        cache_dir = os.path.join(save_dir, f'{cfg.eval.mode}_samples'),
        viz_dir = cfg.eval.viz_dir,
        save_wandb = False,
        return_samples=False,
    )



def eval_autoencoder(cfg, save_dir=None):
    """ Evaluate the Scenario Dreamer AutoEncoder model."""
    model = ScenarioDreamerAutoEncoder(cfg)
    model_summary = ModelSummary(max_depth=-1)
    
    # load checkpoint
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            print("Loading checkpoint: ", ckpt_path)
    
    assert ckpt_path is not None, "No checkpoint found in the save directory."
    
    tester = pl.Trainer(accelerator='auto',
                         devices=1,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_summary],
                         precision='32-true'
                        )
    
    tester.test(model, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    # need to track whether we are evaluating a nuplan or waymo model as 
    # nuplan predicts lane types (lane/green light/red light) and waymo does not
    dataset_name = cfg.dataset_name.name
    if cfg.model_name == 'autoencoder':
        model_name = cfg.model_name
        cfg = cfg.ae
        # not the cleanest solution, but need to track dataset name
        OmegaConf.set_struct(cfg, False)   # unlock to allow setting dataset name
        cfg.dataset_name = dataset_name
        OmegaConf.set_struct(cfg, True)    # relock
    else:
        model_name = cfg.model_name
        cfg_ae = cfg.ae
        cfg = cfg.ldm
        OmegaConf.set_struct(cfg, False)   # unlock to allow setting dataset name
        OmegaConf.set_struct(cfg_ae, False)
        cfg.dataset_name = dataset_name
        cfg_ae.dataset_name = dataset_name
        OmegaConf.set_struct(cfg, True)    # relock
        OmegaConf.set_struct(cfg_ae, True)
    
    pl.seed_everything(cfg.eval.seed, workers=True)

    # checkpoints loaded from here
    save_dir = os.path.join(cfg.eval.save_dir, cfg.eval.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Evaluating Scenario Dreamer {model_name} trained on {cfg.dataset_name} dataset.")

    if model_name == 'autoencoder':
        eval_autoencoder(cfg, save_dir)
    elif model_name == 'ldm':
        if cfg.eval.mode == 'simulation_environments':
            generate_simulation_environments(cfg, cfg_ae, save_dir)
        else:
            eval_ldm(cfg, cfg_ae, save_dir) 


if __name__ == '__main__':
    main()