#!/bin/bash
set -e

# Download HotpotQA (LongRAG ver.)
hf download TIGER-Lab/LongRAG --repo-type dataset --local-dir . --include "hotpot_qa_corpus/*"
rm -rf .cache

# Extract text into .txt files
python extract_text.py
rm -rf hotpot_qa_corpus