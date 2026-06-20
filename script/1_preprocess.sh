#!/bin/bash

trap "echo 'Interrupt received, stopping all processes...'; kill 0" SIGINT

CUDA_DEVICES=(${CUDA_DEVICES:-0 1 2 3})

echo "Beginning preprocessing using CUDA devices: ${CUDA_DEVICES[@]}"

# Query
INPUT_PATH="dataset/query"

OUTPUT_PATH="eval/features/query/qwen3"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[1]} python preprocess/extract_query_feats_qwen3.py --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH"

OUTPUT_PATH="eval/features/query/vlm2vec"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[1]} python preprocess/extract_query_feats_vlm2vec.py --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH"

# Paragraph
INPUT_PATH="dataset/paragraph.parquet"
OUTPUT_PATH="eval/features/paragraph.pkl"
bash script/preprocess/extract_text_feats.sh --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH" --cuda-devices ${CUDA_DEVICES[@]}

# Document
INPUT_PATH="dataset/document.parquet"
OUTPUT_PATH="eval/features/document.pkl"
bash script/preprocess/extract_text_feats.sh --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH" --long-text --cuda-devices ${CUDA_DEVICES[@]}

# Table
INPUT_PATH="dataset/table.parquet"
OUTPUT_PATH="eval/features/table.pkl"
bash script/preprocess/extract_table_feats.sh --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH" --cuda-devices ${CUDA_DEVICES[@]}

# Image
INPUT_PATH="dataset/image.parquet"
OUTPUT_PATH="eval/features/image.pkl"
bash script/preprocess/extract_image_feats.sh --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH" --cuda-devices ${CUDA_DEVICES[@]}

# Clip
INPUT_PATH="dataset/clip.parquet"
OUTPUT_PATH="eval/features/clip.pkl"
bash script/preprocess/extract_clip_feats.sh --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH" --cuda-devices ${CUDA_DEVICES[@]}

# Video
INPUT_PATH="dataset/video.parquet"
OUTPUT_PATH="eval/features/video.pkl"
bash script/preprocess/extract_video_feats.sh --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH" --cuda-devices ${CUDA_DEVICES[@]}
