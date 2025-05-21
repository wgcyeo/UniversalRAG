#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No router specified. Use 'gpt', 't5-large', or 'distilbert'."
    exit 1
fi

if [ "$1" = "gpt" ]; then
    echo "Running GPT routing..."
    python route/gpt/route_gpt.py \
        --input_dir dataset/query \
        --output_dir route/results/gpt

elif [ "$1" = "t5-large" ]; then
    echo "Running T5-Large routing..."
    python route/train/route_t5.py \
        --checkpoint_dir route/train/checkpoints/t5-large \
        --input_dir dataset/query \
        --batch_size 256 \
        --output_dir route/results

elif [ "$1" = "distilbert" ]; then
    echo "Running DistilBERT routing..."
    python route/train/route_distilbert.py \
        --checkpoint_dir route/train/checkpoints/distilbert \
        --input_dir dataset/query \
        --batch_size 256 \
        --output_dir route/results

else
    echo "Error: Unknown router '$1'. Use 'gpt', 't5-large', or 'distilbert'."
    exit 1
fi
