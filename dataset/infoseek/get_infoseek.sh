#!/bin/bash
set -e

# Download InfoSeek dataset
wget http://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0.jsonl.gz
python extract_text.py
rm Wiki6M_ver_1_0.jsonl.gz

hf download ychenNLP/oven --repo-type dataset --local-dir . --include "shard*.tar"
rm -rf .cache
ls shard*.tar 2>/dev/null | parallel -j 8 "tar -xf {} -C ."
rm shard*.tar
mkdir -p images
find 01 02 03 04 06 07 08 09 -type f -exec mv -t images {} +
rm -rf 01 02 03 04 06 07 08 09