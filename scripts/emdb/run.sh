#!/bin/bash

split=2
data_dir="/path/to/emdb"  # Set to your EMDB dataset root (containing P0, P1, ..., P9)
cam_output_dir="results/emdb/camera"
smpl_output_dir="results/emdb/smpl"
eval_input_dir="results/emdb"

python scripts/emdb/run_cam.py --split $split --output_dir "$cam_output_dir" --data_dir "$data_dir"
python scripts/emdb/run_smpl.py --split $split --output_dir "$smpl_output_dir" --data_dir "$data_dir"
python scripts/emdb/run_eval.py --split $split --input_dir "$eval_input_dir" --data_dir "$data_dir"
