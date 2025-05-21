#!/bin/bash

trap "echo 'Interrupt received, stopping all processes...'; kill 0" SIGINT

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <INPUT_PATH> <OUTPUT_PATH> <CUDA_DEVICES (space-separated string)>"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_PATH="$2"
CUDA_DEVICES_STR="$3"
read -a CUDA_DEVICES <<< "$CUDA_DEVICES_STR"

NUM_SPLITS=${#CUDA_DEVICES[@]}

echo -e "\033[34mInput path: $INPUT_PATH | Output path: $OUTPUT_PATH\033[0m"

for ((i = 0; i < NUM_SPLITS; i++)); do
    echo "Processing split $((i + 1))/$NUM_SPLITS on GPU ${CUDA_DEVICES[i]}..."
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[i]} python preprocess/extract_clip_feats.py --input_path $INPUT_PATH --output_path $OUTPUT_PATH --num_splits $NUM_SPLITS --split_index $i --disable_prog &
done

wait

python - <<EOF
import pickle
import glob
import os
import sys

output_path = "$OUTPUT_PATH"
num_splits = $NUM_SPLITS
base, ext = os.path.splitext(output_path)
split_files = sorted(glob.glob(f"{base}_split*{ext}"))

if len(split_files) != num_splits:
    print(f"\033[31mError: Expected {num_splits} split files, but found {len(split_files)}.\033[0m", file=sys.stderr)
    sys.exit(1)

merged_features = {}
for split_file in split_files:
    with open(split_file, 'rb') as f:
        merged_features.update(pickle.load(f))

with open(output_path, 'wb') as f:
    pickle.dump(merged_features, f)

for split_file in split_files:
    os.remove(split_file)
EOF
