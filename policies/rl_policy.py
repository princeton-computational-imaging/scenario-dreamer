from utils.gpudrive_helpers import load_policy

class RLPolicy:
    def __init__(self, cfg):
        self.cfg = cfg

        # Load policy
        self.policy = load_policy(
            path_to_cpt=self.cfg.rl_model_path,
            model_name=self.cfg.rl_model_name,
            device='cuda',
        )

    def act(self, obs):
        actions, _, _, _ = self.policy(obs.float(), deterministic=False)
        return actions