CODE_DIR="$PROJECT_ROOT/data_processing/waymo"

cd "$CODE_DIR"
# uncomment to regenerate the k-disks vocabulary
# python generate_k_disks_vocabulary.py dataset_name=waymo model_name=ctrl_sim
python preprocess_dataset_waymo.py dataset_name=waymo preprocess_waymo.stage=ctrl_sim preprocess_waymo.mode=train preprocess_waymo.chunk_size=12000 preprocess_waymo.num_workers=64
python preprocess_dataset_waymo.py dataset_name=waymo preprocess_waymo.stage=ctrl_sim preprocess_waymo.mode=val preprocess_waymo.chunk_size=12000 preprocess_waymo.num_workers=64