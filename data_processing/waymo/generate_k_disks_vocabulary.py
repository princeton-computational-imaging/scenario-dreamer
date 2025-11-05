import os
import hydra
import numpy as np
import random
import torch
import pickle
from tqdm import tqdm

from cfgs.config import CONFIG_PATH
from datasets.waymo.dataset_ctrl_sim import CtRLSimDataset
from utils.k_disks_helpers import compute_k_disks
from utils.viz import plot_k_disks_vocabulary

SEED = 42
NUM_SCENARIOS = 10000

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    """ Generate k-disks vocabulary from Waymo dataset."""
    project_root = cfg.project_root
    cfg = cfg.ctrl_sim.dataset
    cfg.preprocess = False
    cfg.collect_state_transitions = True
    
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    dset = CtRLSimDataset(cfg, split_name='train')

    state_transitions_all = []
    for idx in tqdm(range(len(dset))):
        with open(dset.files[idx], 'rb') as file:
            data = pickle.load(file)
        
        state_transitions = dset.collect_state_transitions(data)
        state_transitions_all.append(state_transitions)
    
        if idx == NUM_SCENARIOS:
            break 
    state_transitions_all = np.concatenate(state_transitions_all, axis=0)
    
    V = compute_k_disks(
        state_transitions_all, 
        vocab_size=cfg.vocab_size, 
        l=1, 
        w=1, 
        eps=cfg.k_disks_eps
    )
    plot_k_disks_vocabulary(
        V, 
        png_path=os.path.join(
            project_root,
            'metadata',
            f'k_disks_vocab_{cfg.vocab_size}_{cfg.simulation_hz}Hz_seed{SEED}.png'
        ), dpi=1000)
    V_dict = {'V': V}

    with open(os.path.join(
        project_root,
        'metadata',
        f'k_disks_vocab_{cfg.vocab_size}_{cfg.simulation_hz}Hz_seed{SEED}.pkl'
        ), 'wb') as f:
        pickle.dump(V_dict, f)
    print("Finished generating k disks vocabulary.")


if __name__ == "__main__":
    main()