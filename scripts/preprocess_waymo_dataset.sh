CODE_DIR="$PROJECT_ROOT/data_processing/waymo"

cd "$CODE_DIR"
python preprocess_dataset_waymo.py dataset_name=waymo preprocess_waymo.mode=train
python preprocess_dataset_waymo.py dataset_name=waymo preprocess_waymo.mode=val
python preprocess_dataset_waymo.py dataset_name=waymo preprocess_waymo.mode=test
python create_waymo_eval_set.py # create evaluation set (sample 50k real scenes from test set) for computing metrics