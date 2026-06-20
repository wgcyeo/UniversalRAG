#!/bin/bash

cd dataset

BLUE='\033[0;34m'
NC='\033[0m'

# Natural Questions
cd natural_questions
echo -e "${BLUE}Downloading Natural Questions dataset...${NC}"
sh get_nq.sh

# HotpotQA
cd ../hotpotqa
echo -e "${BLUE}Downloading HotpotQA dataset...${NC}"
sh get_hotpotqa.sh

# HybridQA
cd ../HybridQA
echo -e "${BLUE}Downloading HybridQA dataset...${NC}"
sh get_hybridqa.sh

# MRAG-Bench
cd ../MRAG-Bench
echo -e "${BLUE}Downloading MRAG-Bench dataset...${NC}"
sh get_mragbench.sh

# WebQA
cd ../WebQA
echo -e "${BLUE}Downloading WebQA dataset...${NC}"
sh get_webqa.sh

# InfoSeek
cd ../infoseek
echo -e "${BLUE}Downloading InfoSeek dataset...${NC}"
sh get_infoseek.sh

# LVBench
cd ../LVBench
echo -e "${BLUE}Downloading LVBench dataset...${NC}"
sh get_lvbench.sh

# VideoRAG
cd ../videorag
echo -e "${BLUE}Downloading VideoRAG dataset...${NC}"
sh get_videorag.sh

cd ../..
echo -e "${BLUE}Final Processing...${NC}"
python dataset/merge_datasets.py \
    --input-files dataset/natural_questions/nq.parquet dataset/HybridQA/hybridqa_text.parquet dataset/infoseek/infoseek_text.parquet \
    --output-file dataset/paragraph.parquet
rm dataset/natural_questions/nq.parquet dataset/HybridQA/hybridqa_text.parquet dataset/infoseek/infoseek_text.parquet
mv dataset/hotpotqa/hotpotqa.parquet dataset/document.parquet
mv dataset/HybridQA/hybridqa_table.parquet dataset/table.parquet
