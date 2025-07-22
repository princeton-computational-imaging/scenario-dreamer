#!/usr/bin/env bash

CODE_DIR="$PROJECT_ROOT/data_processing/nuplan"

cd "$CODE_DIR"
# ignore all the tf warnings
python generate_nuplan_dataset.py