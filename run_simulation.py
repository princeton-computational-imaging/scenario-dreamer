import hydra
from simulator import Simulator
from policies.idm_policy import IDMPolicy
from policies.rl_policy import RLPolicy
from cfgs.config import CONFIG_PATH

import numpy as np
import torch
import random 
from tqdm import tqdm
from utils.viz import generate_video

class PolicyEvaluator:
    """ Evaluate a given policy in a simulation environment over multiple scenarios."""
    def __init__(self, cfg, policy, env):
        """ Initialize the PolicyEvaluator."""
        self.cfg = cfg
        # policy being evaluated
        self.policy = policy
        # simulation environment
        self.env = env
    
    def reset(self):
        """ Reset the evaluator's statistics and random seeds."""
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        
        self.collision = []
        self.off_route = []
        self.completed = []
        self.progress = []

    
    def update_running_statistics(self, info):
        """ Update running statistics with info from the latest episode."""
        self.collision.append(info['collision'])
        self.off_route.append(info['off_route'])
        self.completed.append(info['completed'])
        self.progress.append(info['progress'])

    
    def compute_metrics(self):
        """ Compute evaluation metrics based on accumulated statistics."""
        metrics_dict = {
            'collision rate': np.array(self.collision).astype(float).mean(),
            'off route rate': np.array(self.off_route).astype(float).mean(),
            'completed rate': np.array(self.completed).astype(float).mean(),
            'progress': np.array(self.progress).astype(float).mean()
        }
        
        return metrics_dict, ["{}: {:.6f}".format(k,v) for (k,v) in metrics_dict.items()]


    def evaluate_policy(self):
        """ Evaluate the policy over all test scenarios in the environment."""
        self.reset()
        
        for i in tqdm(range(self.env.num_test_scenarios)):
            print(f"Simulating environment {i}")
            obs = self.env.reset(i)

            if hasattr(self.policy, 'reset'):
                self.policy.reset(obs)

            for t in range(self.env.steps):
                if self.cfg.visualize:
                    print(f"t={t}")
                    render_frame = True
                    if self.cfg.lightweight:
                        if t%3 != 0:
                            render_frame = False
                    # observations always rendered in local frame of agent
                    if render_frame:
                        self.env.render_state(name=f'{i}', movie_path=self.cfg.movie_path)
                
                action = self.policy.act(obs)
                obs, terminated, info = self.env.step(action)

                if terminated:
                    break

            self.update_running_statistics(info)
            
            if self.cfg.visualize:
                generate_video(name=f'{i}', output_dir=self.cfg.movie_path, delete_images=True)
            
            if self.cfg.verbose:
                if self.cfg.behaviour_model.compute_metrics:
                    print("behaviour model metrics: ", self.env.behaviour_model.compute_metrics()[-1])
                # policy metrics
                print(self.compute_metrics()[-1])

        return self.compute_metrics()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    torch.manual_seed(cfg.sim.seed)
    random.seed(cfg.sim.seed)
    np.random.seed(cfg.sim.seed)

    # initialize simulation environments
    # cfg.sim contains all simulation related configurations
    env = Simulator(cfg)
    
    if cfg.sim.policy == 'rl':
        policy = RLPolicy(cfg.sim)
    else:
        policy = IDMPolicy(cfg, env)
    
    evaluator = PolicyEvaluator(cfg.sim, policy, env)
    _, metrics_str = evaluator.evaluate_policy()
    print(metrics_str)

if __name__ == "__main__":
    main()