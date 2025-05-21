#!/bin/bash
set -e

# Download HowTo100M videos
python download_video.py

# Extract scripts
tar -xzf scripts.tar.gz
rm scripts.tar.gz