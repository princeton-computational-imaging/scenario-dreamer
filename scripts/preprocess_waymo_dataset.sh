CODE_DIR="$PROJECT_ROOT/data_processing/waymo"

cd "$CODE_DIR"
python preprocess_waymo_dataset.py dataset/waymo=autoencoder generate_waymo_dataset.mode=train
python preprocess_waymo_dataset.py dataset/waymo=autoencoder generate_waymo_dataset.mode=val
python preprocess_waymo_dataset.py dataset/waymo=autoencoder generate_waymo_dataset.mode=test