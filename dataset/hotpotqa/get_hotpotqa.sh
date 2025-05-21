#!/bin/bash
set -e

# Download HotpotQA (LongRAG ver.)
python - <<EOF
from huggingface_hub import snapshot_download

repo_id = "TIGER-Lab/LongRAG"
allow_patterns = "hotpot_qa_corpus/*"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=allow_patterns,
    repo_type="dataset",
    local_dir="."
)
EOF
rm -rf .cache

# Extract text into .txt files
python extract_text.py
rm -rf hotpot_qa_corpus