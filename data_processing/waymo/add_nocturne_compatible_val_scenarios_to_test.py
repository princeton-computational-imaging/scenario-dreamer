import os 
import pickle 
import shutil
import hydra
from tqdm import tqdm
from cfgs.config import CONFIG_PATH

# Move half of the nocturne-compatible validation scenarios to the test set to ensure we have 
# a held-out set of nocturne-compatible scenarios for evaluation.
def add_val_to_test(cfg):
    print("Before: ")
    print("Num val scenarios: ", len(os.listdir(cfg.generate_waymo_dataset.output_data_folder_val)))
    print("Num test scenarios: ", len(os.listdir(cfg.generate_waymo_dataset.output_data_folder_test)))
    
    # list containing half of nocturne-compatible filenames in the validation set
    with open(os.path.join(cfg.project_root, 'metadata', 'test_filenames.pkl', 'rb') as f:
        test_filenames = pickle.load(f)

    for filename in tqdm(test_filenames):
        full_filename = 'validation.' + filename + '.pkl'        
        old_path = os.path.join(cfg.generate_waymo_dataset.output_data_folder_val, full_filename)
        new_path = os.path.join(cfg.generate_waymo_dataset.output_data_folder_test, full_filename)

        shutil.move(old_path, new_path)

    print("After: ")
    print("Num val scenarios: ", len(os.listdir(cfg.generate_waymo_dataset.output_data_folder_val)))
    print("Num test scenarios: ", len(os.listdir(cfg.generate_waymo_dataset.output_data_folder_test)))

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    add_val_to_test(cfg)


if __name__ == "__main__":
    main()