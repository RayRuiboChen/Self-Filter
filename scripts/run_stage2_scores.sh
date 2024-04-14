#!/bin/bash

################## VICUNA 7b##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## VICUNA 13b##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-13b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

python self_filter/stage2.py --stage1_model_path ./checkpoints/llava-$MODEL_VERSION-stage1-scores \
    --feature_extractor_setting scores \
    --difficulty_save_name difficulty_scores.json \
    --filtered_annotation_save_path ./data/self_filter_25k_scores.json \
    --filter_num 25000
