import random 
import pickle
import hydra 
from cfgs.config import CONFIG_PATH 
import glob
import os

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    """Cache 50000 random test files for computing metrics on the Waymo dataset."""
    random.seed(42)

    test_dir = os.path.join(cfg.scratch_root, 'scenario_dreamer_ae_preprocess_waymo', 'test')
    test_files = sorted(glob.glob(test_dir + "/*-of-*_*_0_*.pkl")) # grab all lg_type = NON_PARTITIONED files
    random.shuffle(test_files)

    test_files = test_files[:50000]
    test_filenames = [os.path.basename(file) for file in test_files]

    assert len(test_filenames) == 50000

    waymo_test_dict = {
        'files': test_filenames
    }

    with open(os.path.join(cfg.project_root, 'metadata', 'waymo_eval_set.pkl'), 'wb') as f:
        pickle.dump(waymo_test_dict, f)

    print("Done!")

if __name__ == "__main__":
    main()