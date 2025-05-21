#!/bin/bash
set -e

# Download Natural Questions dev
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
gzip -d biencoder-nq-dev.json.gz

# Extract text into .txt files
python extract_text.py
rm biencoder-nq-dev.json.gz