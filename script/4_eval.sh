#!/bin/bash

# Default values
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
ROUTER_MODEL="qwen3-vl-2b"
TARGET="mmlu"
TOP_K=1
ALPHA=0.2
NFRAMES="clip:32,video:32"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model-path PATH      Path to the model checkpoint (default: $MODEL_PATH)"
    echo "                         Choices: Qwen/Qwen3-VL-8B-Instruct, OpenGVLab/InternVL3_5-8B, allenai/Molmo2-4B"
    echo "  --router-model NAME    Router model to use (default: $ROUTER_MODEL)"
    echo "                         Choices: (training-free) gpt-5; (trained) qwen3-vl-2b, internvl3_5-1b, t5gemma-270m"
    echo "  --target NAME          Target dataset for evaluation (default: $TARGET)"
    echo "                         Choices: mmlu, natural_questions, hotpotqa, hybridqa, webqa, mrag_bench, infoseek, lvbench, videorag_wikihow, videorag_synth"
    echo "  --top-k INT            Number of top retrieval to use (default: $TOP_K)"
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
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --router-model) ROUTER_MODEL="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
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
    --model-path "$MODEL_PATH" \
    --router-model "$ROUTER_MODEL" \
    --target "$TARGET" \
    --top-k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES"
