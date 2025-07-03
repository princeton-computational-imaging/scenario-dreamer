CODE_DIR="$PROJECT_ROOT/data_processing/waymo"

cd "$CODE_DIR"
python preprocess_waymo_dataset.py dataset/waymo=autoencoder preprocess_waymo.mode=train
python preprocess_waymo_dataset.py dataset/waymo=autoencoder preprocess_waymo.mode=val
python preprocess_waymo_dataset.py dataset/waymo=autoencoder preprocess_waymo.mode=test