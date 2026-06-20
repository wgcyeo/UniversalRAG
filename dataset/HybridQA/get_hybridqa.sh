#!/bin/bash
set -e

# Download HybridQA WikiTables-WithLinks dataset
git clone https://github.com/wenhuchen/WikiTables-WithLinks

# Extract tables and text into parquet files
python extract_table.py
python extract_text.py
rm -rf WikiTables-WithLinks