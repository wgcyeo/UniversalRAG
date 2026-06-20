#!/bin/bash

trap "echo 'Interrupt received, stopping all processes...'; kill 0" SIGINT

# Default values
INPUT_PATH=""
OUTPUT_PATH=""
CUDA_DEVICES=()

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --cuda-devices)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CUDA_DEVICES+=("$1")
                shift
            done
            ;;
        -h|--help)
            echo "Usage: $0 --input-path <path> --output-path <path> --cuda-devices <device1> [device2] ..."
            echo ""
            echo "Options:"
            echo "  --input-path           Path to parquet file with clip_id, frame_num, and frame_time columns"
            echo "  --output-path          Path to output pickle file"
            echo "  --cuda-devices         Space-separated list of CUDA device IDs"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_PATH" ]; then
    echo "Error: --input-path is required"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: --output-path is required"
    exit 1
fi

if [ ${#CUDA_DEVICES[@]} -eq 0 ]; then
    echo "Error: --cuda-devices requires at least one device"
    exit 1
fi

NUM_SPLITS=${#CUDA_DEVICES[@]}

echo -e "\033[34mInput path: $INPUT_PATH | Output path: $OUTPUT_PATH\033[0m"

LOG_DIR=".log"
mkdir -p "$LOG_DIR"
TIME_INDEX=$(date +%Y%m%d_%H%M%S)

for ((i = 0; i < NUM_SPLITS; i++)); do
    LOG_FILE="$LOG_DIR/extract_clip_feats_split_${i}_${TIME_INDEX}.log"
    echo "Processing split $((i + 1))/$NUM_SPLITS on GPU ${CUDA_DEVICES[i]}... (logging to $LOG_FILE)"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[i]} python preprocess/extract_clip_feats.py --input-path $INPUT_PATH --output-path $OUTPUT_PATH --num-splits $NUM_SPLITS --split-index $i > "$LOG_FILE" 2>&1 &
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
