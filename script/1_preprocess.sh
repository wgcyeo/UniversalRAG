#!/bin/bash

trap "echo 'Interrupt received, stopping all processes...'; kill 0" SIGINT

CUDA_DEVICES=(0 1 2 3)
CUDA_DEVICES_STR="${CUDA_DEVICES[@]}"

echo "Beginning preprocessing using CUDA devices: ${CUDA_DEVICES[@]}"

# Query

INPUT_PATH="dataset/query"

OUTPUT_PATH="eval/features/query/internvideo"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[0]} python preprocess/extract_query_feats_internvideo.py --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH"

OUTPUT_PATH="eval/features/query/bge-large"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[0]} python preprocess/extract_query_feats_bge.py --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH"

# SQuAD

INPUT_PATH="dataset/squad/text/"
OUTPUT_PATH="eval/features/text/squad.pkl"
LONG_TEXT=false

bash script/preprocess/extract_text_feats.sh "$INPUT_PATH" "$OUTPUT_PATH" "$LONG_TEXT" "$CUDA_DEVICES_STR"

# Natural Questions

INPUT_PATH="dataset/natural_questions/text/"
OUTPUT_PATH="eval/features/text/natural_questions.pkl"
LONG_TEXT=false

bash script/preprocess/extract_text_feats.sh "$INPUT_PATH" "$OUTPUT_PATH" "$LONG_TEXT" "$CUDA_DEVICES_STR"

# HotpotQA

INPUT_PATH="dataset/hotpotqa/text/"
OUTPUT_PATH="eval/features/text/hotpotqa.pkl"
LONG_TEXT=true

bash script/preprocess/extract_text_feats.sh "$INPUT_PATH" "$OUTPUT_PATH" "$LONG_TEXT" "$CUDA_DEVICES_STR"

# WebQA

INPUT_PATH="dataset/WebQA/webqa_images.json"
OUTPUT_PATH="eval/features/image/webqa.pkl"

bash script/preprocess/extract_image_feats.sh "$INPUT_PATH" "$OUTPUT_PATH" "$CUDA_DEVICES_STR"

# LVBench

INPUT_PATH="eval/features/clip/lvbench_clipframenum.pkl"
OUTPUT_PATH="eval/features/clip/lvbench.pkl"

bash script/preprocess/extract_clip_feats.sh "$INPUT_PATH" "$OUTPUT_PATH" "$CUDA_DEVICES_STR"

INPUT_PATH="dataset/LVBench/videos/"
OUTPUT_PATH="eval/features/video/lvbench.pkl"

bash script/preprocess/extract_video_feats.sh "$INPUT_PATH" "$OUTPUT_PATH" "$CUDA_DEVICES_STR"

INPUT_PATH="eval/features/clip/lvbench_clipframetime.pkl"
OUTPUT_PATH="eval/features/clip/lvbench_clipscript.pkl"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[0]} python preprocess/extract_clipscript_feats.py --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH"

INPUT_PATH="dataset/LVBench/"
OUTPUT_PATH="eval/features/video/lvbench_vidscript.pkl"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[0]} python preprocess/extract_vidscript_feats.py --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH"

# HowTo100M

INPUT_PATH="eval/features/clip/howto100m_clipframenum.pkl"
OUTPUT_PATH="eval/features/clip/howto100m.pkl"

bash script/preprocess/extract_clip_feats.sh "$INPUT_PATH" "$OUTPUT_PATH" "$CUDA_DEVICES_STR"

INPUT_PATH="dataset/videorag/videos/"
OUTPUT_PATH="eval/features/video/howto100m.pkl"

bash script/preprocess/extract_video_feats.sh "$INPUT_PATH" "$OUTPUT_PATH" "$CUDA_DEVICES_STR"

INPUT_PATH="dataset/videorag/"
OUTPUT_PATH="eval/features/video/videorag_vidscript.pkl"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[0]} python preprocess/extract_vidscript_feats.py --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH"
