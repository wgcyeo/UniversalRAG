#!/bin/bash

cd dataset

BLUE='\033[0;34m'
NC='\033[0m'

# SQuAD
cd squad
echo -e "${BLUE}Downloading SQuAD dataset...${NC}"
sh get_squad.sh

# Natural Questions
cd ../natural_questions
echo -e "${BLUE}Downloading Natural Questions dataset...${NC}"
sh get_nq.sh

# HotpotQA
cd ../hotpotqa
echo -e "${BLUE}Downloading HotpotQA dataset...${NC}"
sh get_hotpotqa.sh

# WebQA
cd ../WebQA
echo -e "${BLUE}Downloading WebQA dataset...${NC}"
sh get_webqa.sh

# LVBench
cd ../LVBench
echo -e "${BLUE}Downloading LVBench dataset...${NC}"
sh get_lvbench.sh

# VideoRAG
cd ../videorag
echo -e "${BLUE}Downloading VideoRAG dataset...${NC}"
sh get_videorag.sh

cd ../..
