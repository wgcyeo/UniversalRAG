#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No model specified. Use 't5-large' or 'distilbert'."
    exit 1
fi

if [ "$1" = "t5-large" ]; then
    echo "Running T5-Large training..."
    python route/train/train_t5.py \
        --model_name google/flan-t5-large \
        --input_dir route/train/data/train_data_t5.json \
        --num_train_epochs 10 \
        --learning_rate 3e-5 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --checkpoint_dir route/train/checkpoints/

elif [ "$1" = "distilbert" ]; then
    echo "Running DistilBERT training..."
    python route/train/train_distilbert.py \
        --model_name distilbert-base-uncased \
        --input_dir route/train/data/train_data_distilbert.json \
        --num_train_epochs 5 \
        --learning_rate 2e-5 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --checkpoint_dir route/train/checkpoints/

else
    echo "Error: Unknown model '$1'. Use 't5-large' or 'distilbert'."
    exit 1
fi
