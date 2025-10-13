#!/bin/bash

# Nebula Dataset Fine-tuning Script for 2 GPUs
# Usage: bash run_nebula_finetune.sh

set -e  # Exit on any error

# Configuration
export CUDA_VISIBLE_DEVICES=1,2
export LAUNCHER="pytorch"  # Set launcher to pytorch instead of slurm
export TOKENIZERS_PARALLELISM=true
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# CPU affinity (optional - adjust based on your system)
CPU_CORES="8-15"

# Training configuration
CONFIG_FILE="nebula_finetune_cfg.json"
NUM_GPUS=2
MASTER_PORT=29500 #29500  # Change if port is in use

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "Starting distributed training on $NUM_GPUS GPUs..."
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config file: $CONFIG_FILE"
echo "CPU cores: $CPU_CORES"

# Run distributed training using torchrun
taskset -c $CPU_CORES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    spatialvla_finetune_nebula.py \
    $CONFIG_FILE

echo "Training completed!"