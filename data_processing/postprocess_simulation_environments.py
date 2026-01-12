import os
import pickle
from tqdm import tqdm
import numpy as np
import hydra

from utils.sim_env_helpers import postprocess_sim_env
from cfgs.config import CONFIG_PATH


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    dataset_name = cfg.dataset_name.name
    pre_path = cfg.postprocess_sim_envs.pre_path
    post_path = cfg.postprocess_sim_envs.post_path
    route_length = cfg.postprocess_sim_envs.route_length
    max_num_envs = cfg.postprocess_sim_envs.max_num_envs

    if max_num_envs == -1:
        max_num_envs = len(os.listdir(pre_path))
    
    os.makedirs(post_path, exist_ok=True)

    num_sim_envs = 0
    for i, filename in enumerate(tqdm(os.listdir(pre_path))):
        file_path = os.path.join(pre_path, filename)

        with open(file_path, "rb") as f:
            sim_env = pickle.load(f)
        
        # Filter to only lane-based scenarios (for nuPlan)
        if dataset_name == 'nuplan' and sim_env['lane_types'][:, 0].sum() < len(sim_env['lane_types']):
            continue # skip if not all lane types are lane

        sim_env_filtered = postprocess_sim_env(
            sim_env,
            route_length,
            dataset_name)

        if sim_env_filtered['route_lane_indices'] is None:
            continue

        with open(os.path.join(post_path, filename), "wb") as f:
            pickle.dump(sim_env_filtered, f)
        num_sim_envs += 1

        if num_sim_envs >= max_num_envs:
            break

    print(f"Post-processed {num_sim_envs} simulation environments saved to {post_path}")


if __name__ == '__main__':
    main()
    

    

    

