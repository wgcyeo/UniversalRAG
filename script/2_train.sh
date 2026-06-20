#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No model specified. Use 't5gemma-270m', 'qwen3-vl-2b', or 'internvl3_5-1b'."
    exit 1
fi

echo "Preparing training data..."
python route/train/prep_train_data.py

if [ "$1" = "qwen3-vl-2b" ]; then
    echo "Running Qwen3-VL-2B training..."
    python route/train/train_qwen3_vl.py \
        --model-name Qwen/Qwen3-VL-2B-Instruct \
        --train-data route/train/data/train_data.json \
        --output-dir route/train/checkpoints/qwen3_vl_2b

elif [ "$1" = "internvl3_5-1b" ]; then
    echo "Running InternVL3.5-1B training..."
    python route/train/train_internvl3_5.py \
        --model-name OpenGVLab/InternVL3_5-1B \
        --train-data route/train/data/train_data.json \
        --output-dir route/train/checkpoints/internvl3_5_1b

elif [ "$1" = "t5gemma-270m" ]; then
    echo "Running T5Gemma-270M training..."
    python route/train/train_t5gemma.py \
        --model-name google/t5gemma-2-270m-270m \
        --train-data route/train/data/train_data.json \
        --output-dir route/train/checkpoints/t5gemma_270m

else
    echo "Error: Unknown model '$1'. Use 'qwen3-vl-2b', 'internvl3_5-1b', or 't5gemma-270m'"
    exit 1
fi
