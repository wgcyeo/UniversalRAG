#!/bin/bash
set -e

# Download LVBench videos
python download_video.py

# Extract scripts
tar -xzf scripts.tar.gz
rm scripts.tar.gz