#!/bin/bash
set -e

# Download MRAG-Bench dataset
hf download uclanlp/MRAG-Bench --repo-type dataset --local-dir . --include "data/*"
rm -rf .cache

# Extract images into .jpg files
python extract_image.py
rm -rf data