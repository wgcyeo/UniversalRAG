#!/bin/bash
set -e

# Download SQuAD 1.1 dev
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz
gzip -d biencoder-squad1-dev.json.gz

# Extract text into .txt files
python extract_text.py
rm biencoder-squad1-dev.json