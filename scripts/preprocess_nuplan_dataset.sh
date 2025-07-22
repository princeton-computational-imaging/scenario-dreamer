CODE_DIR="$PROJECT_ROOT/data_processing/nuplan"

cd "$CODE_DIR"
python preprocess_dataset_nuplan.py dataset_name=nuplan preprocess_nuplan.mode=train
python preprocess_dataset_nuplan.py dataset_name=nuplan preprocess_nuplan.mode=val
python preprocess_dataset_nuplan.py dataset_name=nuplan preprocess_nuplan.mode=test