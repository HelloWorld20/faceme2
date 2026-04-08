#!/bin/bash

# ==============================================================================
# FaceMe2 Dual-Branch (SwinIR + ArcFace) Optimized Training Script
# ==============================================================================
# This script is based on the original parameters provided in scripts.md

# Set custom cache directories to avoid "No space left on device" in /home
export TORCH_HOME="/data/weijianghong/.cache/torch"
export HF_HOME="/data/weijianghong/.cache/huggingface"
export XDG_CACHE_HOME="/data/weijianghong/.cache"

# Ensure the cache directories exist
mkdir -p $TORCH_HOME
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

echo "Starting FaceMe2 Optimized Training..."

CUDA_VISIBLE_DEVICES=1,2,3,4,5 accelerate launch --num_processes=5 train.py \
 --pretrained_model_name_or_path "/data/weijianghong/workspace/faceme2/models/RealVisXL_V3.0" \
 --mix_pretrained_path "None" \
 --output_dir "./output/train_results" \
 --train_data_dir "output/train_json/train.json" \
 --resolution 256 \
 --report_to "wandb" \
 --learning_rate 5e-5 \
 --train_batch_size 1 \
 --mixed_precision fp16 \
 --num_workers 4 \
 --gradient_accumulation_steps 2 \
 --num_train_epochs 100 \
 --checkpoint_steps 1000 \
 --max_train_samples 1000 \
 --exp_name "faceme2_dual_branch"

echo "Training command executed."