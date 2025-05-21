#!/bin/bash

# Default values
MODEL_PATH="OpenGVLab/InternVL2_5-8B"
ROUTER_MODEL="distilbert"
TARGET="mmlu"
TOP_K=1
ALPHA=0.2
NFRAMES="clip:32,video:32"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_path PATH      Path to the model checkpoint (default: $MODEL_PATH)"
    echo "                         Choices: OpenGVLab/InternVL2_5-8B, Qwen/Qwen2.5-VL-7B-Instruct, microsoft/Phi-3.5-vision-instruct"
    echo "  --router_model NAME    Router model to use (default: $ROUTER_MODEL)"
    echo "                         Choices: gpt, t5-large, distilbert"
    echo "  --target NAME          Target dataset for evaluation (default: $TARGET)"
    echo "                         Choices: mmlu, squad, natural_questions, hotpotqa, webqa, lvbench, videorag_wikihow, videorag_synth"
    echo "  --top_k INT            Number of top retrieval to use (default: $TOP_K)"
    echo "  --alpha FLOAT          Weight for image caption or clip/video script features (default: $ALPHA, range: 0 to 1)"
    echo "  --nframes STR          Number of frames to process for each modality (default: $NFRAMES)"
    echo "                         Example: 'clip:8,video:32'"
    echo "  -h, --help             Show this help message and exit"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --router_model) ROUTER_MODEL="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --top_k) TOP_K="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        --nframes) NFRAMES="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage."
            exit 1
            ;;
    esac
done

python eval/eval.py \
    --model_path "$MODEL_PATH" \
    --router_model "$ROUTER_MODEL" \
    --target "$TARGET" \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES"
