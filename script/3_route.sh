#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No router specified. Use 'gpt-5', 'qwen3-vl-2b', 'internvl3_5-1b', or 't5gemma-270m'."
    exit 1
fi

if [ "$1" = "gpt-5" ]; then
    echo "Running GPT-5 routing..."
    python route/gpt/route_gpt.py \
        --model-name gpt-5 \
        --input-dir dataset/query \
        --output-dir route/results/gpt-5

elif [ "$1" = "qwen3-vl-2b" ]; then
    echo "Running Qwen3-VL-2B (trained) routing..."
    python route/train/route_qwen3_vl.py \
        --model-path route/train/checkpoints/qwen3_vl_2b \
        --input-dir dataset/query \
        --batch-size 128 \
        --output-dir route/results/qwen3_vl_2b

elif [ "$1" = "internvl3_5-1b" ]; then
    echo "Running InternVL3.5-1B (trained) routing..."
    python route/train/route_internvl3_5.py \
        --model-path route/train/checkpoints/internvl3_5_1b \
        --input-dir dataset/query \
        --batch-size 128 \
        --output-dir route/results/internvl3_5_1b

elif [ "$1" = "t5gemma-270m" ]; then
    echo "Running T5Gemma-270M (trained) routing..."
    python route/train/route_t5gemma.py \
        --model-path route/train/checkpoints/t5gemma_270m \
        --input-dir dataset/query \
        --batch-size 256 \
        --output-dir route/results/t5gemma_270m

else
    echo "Error: Unknown router '$1'. Use 'gpt-5', 'qwen3-vl-2b', 'internvl3_5-1b', or 't5gemma-270m'."
    exit 1
fi
