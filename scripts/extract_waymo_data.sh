CODE_DIR="$PROJECT_ROOT/data_processing/waymo"

cd "$CODE_DIR"
python generate_waymo_dataset.py generate_waymo_dataset.mode=train
python generate_waymo_dataset.py generate_waymo_dataset.mode=val
python generate_waymo_dataset.py generate_waymo_dataset.mode=test
python add_nocturne_compatible_val_scenarios_to_test.py 